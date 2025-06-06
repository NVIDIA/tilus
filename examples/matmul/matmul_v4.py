import math

import pandas
import pandas as pd
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func

tilus.option.cache_dir("./cache")

pd.set_option("display.float_format", lambda x: "%.2f" % x)


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128)])
@tilus.autotune("block_k", [16, 32])
class MatmulV4(tilus.Script):
    debug_schedule = dict(
        num_warps=4,
        block_m=128,
        block_n=64,
        block_k=16,
    )

    def __init__(
        self,
        num_warps,
        block_m,
        block_n,
        block_k,
    ):
        super().__init__()
        self.mma = self.cuda.resolve_dot_config(float16, float32, num_warps=num_warps, m=block_m, n=block_n, k=block_k)
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps

        self.layout_sa = self.cuda.swizzled_shared_layout(dtype=float16, shape=[self.block_m, self.block_k])
        self.layout_sb = self.cuda.swizzled_shared_layout(dtype=float16, shape=[self.block_k, self.block_n])

    def __call__(self, m_size: int32, n_size: int, k_size: int, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = [self.utils.ceil_div(m_size, self.block_m), self.utils.ceil_div(n_size, self.block_n)]
        self.attrs.warps = self.num_warps

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, layout=self.layout_sa)
        sb = self.shared_tensor(dtype=float16, layout=self.layout_sb)
        acc = self.register_tensor(dtype=float32, layout=self.mma.lc, init=0.0)

        for offset_k in range(0, k_size, block_k):
            self.copy_async(src=ga, dst=sa, offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb, offsets=[offset_k, offset_n])
            self.copy_async_wait_all()
            self.sync()

            a = self.load_shared(sa, layout=self.mma.la)
            b = self.load_shared(sb, layout=self.mma.lb)
            self.mma_dot(a, b, acc, output=acc)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [2048, 2048, 2048],
        [4096, 4096, 4096],
        [4097, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV4()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        matmul(m, n, k, a, b, c_actual)

        # check correctness
        torch.testing.assert_close(c_expect, c_actual)

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


if __name__ == "__main__":
    main()
