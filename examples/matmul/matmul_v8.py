import math

import pandas
import pandas as pd
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")
tilus.option.debug.dump_ir()
tilus.utils.clear_cache()

pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128), (32, 256)])
@tilus.autotune("block_k", [16, 32])
@tilus.autotune("num_stages", [3, 4, 5])
@tilus.autotune("split_k_factor", [1, 4, 12, 16])
class MatmulV7(tilus.Script):
    debug_schedule = dict(
        num_warps=4,
        block_m=64,
        block_n=128,
        block_k=16,
        num_stages=4,
        split_k_factor=1,
    )

    def __init__(
        self,
        num_warps,
        block_m,
        block_n,
        block_k,
        num_stages,
        split_k_factor,
    ):
        super().__init__()
        self.mma = self.cuda.default_dot_config(float16, float32, num_warps=num_warps, m=block_m, n=block_n, k=block_k)
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor

        self.layout_sa = self.cuda.swizzled_shared_layout(float16, shape=[num_stages, self.block_m, self.block_k])
        self.layout_sb = self.cuda.swizzled_shared_layout(float16, shape=[num_stages, self.block_k, self.block_n])
        self.layout_sc = self.cuda.swizzled_shared_layout(float16, shape=[self.block_m, self.block_n])

    def __call__(self, m_size: int32, n_size: int, k_size: int, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n), self.split_k_factor]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, layout=self.layout_sa)
        sb = self.shared_tensor(dtype=float16, layout=self.layout_sb)
        acc = self.register_tensor(dtype=float32, layout=self.mma.lc, init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(start_offset_k, end_offset_k, block_k, unroll=self.num_stages):
            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            if preload_offset_k < end_offset_k:
                self.copy_async(src=ga, dst=sa[preload_stage], offsets=[offset_m, preload_offset_k])
                self.copy_async(src=gb, dst=sb[preload_stage], offsets=[preload_offset_k, offset_n])
            self.copy_async_commit_group()

            # computation for current tile
            a = self.load_shared(sa[current_stage], layout=self.mma.la)
            b = self.load_shared(sb[current_stage], layout=self.mma.lb)
            self.mma_dot(a, b, acc, output=acc)

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.copy_async_wait_group(n=self.num_stages - 2)
            self.sync()

        # there might be on-fly copy_async in the pipeline, we need to wait for all of them
        self.free_shared(sa)
        self.free_shared(sb)

        # cast the accumulator to float16 and change the register tensor's layout
        sc = self.shared_tensor(dtype=float16, layout=self.layout_sc)
        casted_acc = self.cast(acc, dtype=float16)
        self.store_shared(sc, casted_acc)
        self.sync()
        rc = self.load_shared(sc)
        self.free_shared(sc)

        m_blocks, n_blocks = cdiv(m_size, block_m), cdiv(n_size, block_n)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        if self.split_k_factor == 0:
            self.store_global(gc, rc, offsets=[offset_m, offset_n])
        else:
            semaphores = self.global_tensor(dtype=int32, shape=[m_blocks, n_blocks], requires_clean=True)
            semaphore: ~int32 = ~semaphores[self.blockIdx.x, self.blockIdx.y]

            # load and accumulate the partial result in global memory
            if self.blockIdx.z > 0:
                self.lock_semaphore(semaphore, value=self.blockIdx.z)
                partial_rc = self.load_global(gc, offsets=[offset_m, offset_n], shape=[block_m, block_n])
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor)


def main():
    torch.random.manual_seed(41)
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = []
    for k, n in [
        [4096, 4096 * 3],
        [4096, 4096],
        [4096, 14336 * 2],
        [14336, 4096],
    ]:
        for m in [
            1,
            16,
            32,
            4096,
            4097,
        ]:
            workloads.append([m, n, k])

    rows = []
    matmul = MatmulV7()
    for m, n, k in workloads:
        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        matmul(m, n, k, a, b, c_actual)

        # check correctness
        torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)

        # benchmark
        for name, func in [
            ("torch", lambda: torch.matmul(a, b, out=c_expect)),
            ("tilus", lambda: matmul(m, n, k, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            flops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, flops])

        # Create initial DataFrame
        df = pandas.DataFrame(rows, columns=headers)

        # Post-process to combine torch and tilus results
        df_pivot = df.pivot(index=["m", "n", "k"], columns="name", values=["latency (ms)", "gflops"])
        df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
        df_pivot = df_pivot.reset_index()

        # Calculate speedup (torch latency / tilus latency)
        df_pivot["speedup"] = df_pivot["torch_latency (ms)"] / df_pivot["tilus_latency (ms)"]

        # Reorder columns for better readability
        column_order = [
            "m",
            "n",
            "k",
            "torch_latency (ms)",
            "torch_gflops",
            "tilus_latency (ms)",
            "tilus_gflops",
            "speedup",
        ]
        df_pivot = df_pivot[column_order]

        print(df_pivot)


if __name__ == "__main__":
    main()
