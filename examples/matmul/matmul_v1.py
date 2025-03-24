import math

import pandas
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func

tilus.option.cache_dir("./cache")


class MatmulV1(tilus.Script):
    def __init__(self):
        super().__init__()
        self.mma = self.cuda.mma.m16n8k16_f16_f32
        self.block_m = self.mma.m
        self.block_n = self.mma.n
        self.block_k = self.mma.k

    def __call__(self, m_size: int32, n_size: int, k_size: int, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = [self.utils.ceil_div(m_size, self.block_m), self.utils.ceil_div(n_size, self.block_n)]
        self.attrs.warps = 1

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.block_k, self.block_n])
        acc = self.register_tensor(dtype=float32, layout=self.mma.lc, init=0.0)

        for offset_k in range(0, k_size, self.block_k):
            lda = self.load_global(ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k])
            self.store_shared(sa, lda)
            ldb = self.load_global(gb, offsets=[offset_k, offset_n], shape=[self.block_k, self.block_n])
            self.store_shared(sb, ldb)
            self.sync()

            a = self.load_shared(sa, out_layout=self.mma.la)
            b = self.load_shared(sb, out_layout=self.mma.lb)
            acc = self.mma_dot(a, b, acc, config=self.mma, warp_spatial=(1, 1, 1), warp_repeat=(1, 1, 1))
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [1025, 1025, 1026],
        [2048, 2048, 2048],
        [4096, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV1()

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

    pandas.set_option("display.float_format", lambda x: "%.2f" % x)
    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
