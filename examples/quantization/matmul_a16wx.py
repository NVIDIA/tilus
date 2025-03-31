import math

import pandas
import tilus
import torch
from tilus import Script, float16, float32, int32, uint8
from tilus.ir.layout import reduce, repeat, spatial
from tilus.utils import benchmark_func, cdiv, gcd

from hidet import bfloat16
from hidet.ir.type import DataType, void_p

tilus.option.cache_dir("./cache")


class QuantizedMatmulCommon(Script):
    def __init__(self, weight_tile: tuple[int, int], a_dtype: DataType, b_dtype: DataType):
        super().__init__()
        tile_k, tile_n = weight_tile
        assert a_dtype in [float16, bfloat16], "this kernel only supports float16/bfloat16 as activation data type"
        if a_dtype == float16:
            self.mma = self.cuda.mma_configs.m16n8k16_f16_f32
        else:
            self.mma = self.cuda.mma_configs.m16n8k16_bf16_f32
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.tile_k = weight_tile[0]
        self.tile_n = weight_tile[1]
        self.tile_layout = repeat(tile_k // self.mma.k, tile_n // self.mma.n) * self.mma.lb

        bits_per_threads = self.tile_layout * b_dtype.nbits

        assert bits_per_threads % 8 == 0, "bits_per_threads must be divisible by 8"

        bytes_per_threads = bits_per_threads // 8

        # view as bytes
        inner_size = gcd(bytes_per_threads, 16)
        outer_size = bytes_per_threads // inner_size

        self.flatten_tile_layout = repeat(outer_size).spatial(32).repeat(inner_size)


class QuantizedMatmulChangeLayout(QuantizedMatmulCommon):
    def __init__(self, weight_tile: tuple[int, int], group_size: int, a_dtype: DataType, b_dtype: DataType):
        super().__init__(weight_tile=weight_tile, a_dtype=a_dtype, b_dtype=b_dtype)
        self.weight_tile = weight_tile
        self.group_size = group_size
        self.dtype = b_dtype
        self.layout_src = self.tile_layout
        self.layout_dst = self.flatten_tile_layout

    def __call__(self, n_size: int, k_size: int, src_ptr: void_p, dst_ptr: void_p):
        self.static_assert(n_size % self.tile_n == 0, "n_size must be divisible by tile_n")
        self.static_assert(k_size % self.tile_k == 0, "k_size must be divisible by tile_k")
        self.attrs.warps = 1
        self.attrs.blocks = [k_size // self.tile_k, n_size // self.tile_n]

        offset_k = self.blockIdx.x * self.tile_k
        offset_n = self.blockIdx.y * self.tile_n

        g_src = self.global_view(src_ptr, dtype=self.dtype, shape=[k_size, n_size])
        r_src = self.load_global(g_src, offsets=[offset_k, offset_n], layout=self.layout_src)
        r_dst = self.view(r_src, layout=self.layout_dst, dtype=uint8)
        g_dst = self.global_view(
            dst_ptr, dtype=uint8, shape=[k_size // self.tile_k, n_size // self.tile_n, self.layout_dst.shape[0]]
        )
        self.store_global(g_dst, r_dst, offsets=[self.blockIdx.x, self.blockIdx.y])


class QuantizedMatmul(QuantizedMatmulCommon):
    def __init__(
        self,
        weight_tile: tuple[int, int],
        group_size: int,
        a_dtype: DataType,
        b_dtype: DataType,
        warp_spatial: tuple[int, int],
        warp_repeat: tuple[int, int, int],
        num_stages: int,
        split_k_factor: int,
    ):
        super().__init__(weight_tile=weight_tile, a_dtype=a_dtype, b_dtype=b_dtype)

        assert a_dtype.is_any_float16(), "this kernel only supports float16/bfloat16 as activation data type"
        assert 1 <= b_dtype.nbits <= 8, "this kernel only supports dtype with 1-8 bits as weight data type"

        self.weight_tile = weight_tile
        self.group_size = group_size
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.warp_spatial = warp_spatial
        self.warp_repeat = warp_repeat
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor

        wsm, wsn = warp_spatial
        wrm, wrn, wrk = warp_repeat
        tk, tn = weight_tile[0] // self.mma.k, weight_tile[1] // self.mma.n

        assert wrk % tk == 0 and wrn % tn == 0

        self.block_m = self.mma.m * wsm * wrm
        self.block_n = self.mma.n * wsn * wrn
        self.block_k = self.mma.k * wrk
        self.num_warps = wsm * wsn

        k_tiles = wrk // tk
        n_tiles = wsn * wrn // tn
        self.tile_bytes = tk * tn * self.mma.lb.local_size * b_dtype.nbits // 8

        # we make sure that each weight_tile will be loaded by one warp
        assert wrk * self.mma.k % weight_tile[0] == 0
        assert wrn * self.mma.n % weight_tile[1] == 0

        # [block_m, block_k]
        self.layout_ra = reduce(spatial(wsm, 1, wsn, ranks=[1, 0, 2]), dims=[2]).repeat(wrm, wrk) * self.mma.la

        # [k_tiles, n_tiles]
        layout_rb_head = reduce(spatial(1, wsn, wsm, ranks=[0, 2, 1]), dims=[2]).repeat(wrk // tk, wrn // tn)

        # [k_tiles, n_tiles, tile_bytes]
        self.layout_rb_flattened = layout_rb_head + self.flatten_tile_layout

        # [block_k, block_n] = [tiles_k * tk * mma.k, tiles_n * tn * mma.n]
        self.layout_rb = layout_rb_head * self.tile_layout

        # [block_m, block_n]
        self.layout_rc = spatial(wsm, wsn).repeat(wrm, wrn) * self.mma.lc

        self.layout_sa = self.cuda.swizzled_shared_layout(
            dtype=self.a_dtype, bs=num_stages, m=self.block_m, n=self.block_k
        )
        self.layout_sb = self.cuda.shared_layout(shape=[self.num_stages, k_tiles, n_tiles, self.tile_bytes])
        self.layout_sc = self.cuda.swizzled_shared_layout(dtype=self.a_dtype, m=self.block_m, n=self.block_n)

    def __call__(
        self, m_size: int32, n_size: int, k_size: int, a_ptr: void_p, b_ptr: void_p, scale_ptr: void_p, c_ptr: void_p
    ):
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n), self.split_k_factor]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=self.a_dtype, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=uint8, shape=[k_size // self.tile_k, n_size // self.tile_n, self.tile_bytes])
        sa = self.shared_tensor(dtype=float16, layout=self.layout_sa)
        sb = self.shared_tensor(dtype=uint8, layout=self.layout_sb)
        acc = self.register_tensor(dtype=float32, layout=self.layout_rc, init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k // self.tile_k, offset_n // self.tile_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(start_offset_k, end_offset_k, block_k, unroll=self.num_stages):
            # computation for current tile
            a = self.load_shared(sa[current_stage], out_layout=self.layout_ra)
            b_flattened = self.load_shared(sb[current_stage], out_layout=self.layout_rb_flattened)
            b_low_precision = self.view(b_flattened, dtype=self.b_dtype, layout=self.layout_rb)
            b = self.cast(b_low_precision, dtype=self.a_dtype)
            acc = self.mma_dot(a, b, acc, config=self.mma, warp_spatial=self.warp_spatial, warp_repeat=self.warp_repeat)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            self.copy_async(src=ga, dst=sa[preload_stage], offsets=[offset_m, preload_offset_k])
            self.copy_async(
                src=gb, dst=sb[preload_stage], offsets=[preload_offset_k // self.tile_k, offset_n // self.tile_n]
            )
            self.copy_async_commit_group()
            self.copy_async_wait_group(n=self.num_stages - 2)

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.sync()

        # there might be on-fly copy_async in the pipeline, we need to wait for all of them
        self.copy_async_wait_all()
        self.sync()
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
                partial_rc = self.load_global(gc, offsets=[offset_m, offset_n], layout=rc.layout)
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor)


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [2048, 2048, 2048],
        [4096, 4096, 4096],
        [4097, 4096, 4096],
        [1, 4096, 4096],
        [2, 4096, 4096],
        [3, 4096, 4096],
        [16, 4096, 4096],
        [32, 4096, 4096],
    ]

    rows = []
    matmul = QuantizedMatmul()
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

        df = pandas.DataFrame(rows, columns=headers)
        print(df)


# if __name__ == "__main__":
#     main()
