import math

import pytest
import tilus
import torch
from tilus import float16, float32, int32


class MatmulV1(tilus.Script):
    def __init__(self, n_size: int, k_size: int):
        super().__init__()
        self.mma = self.cuda.mma.m16n8k16_f16_f32
        self.n_size = n_size
        self.k_size = k_size

        self.block_m = self.mma.m
        self.block_n = self.mma.n
        self.block_k = self.mma.k

    def kernel(self, m_size: int32, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = [self.utils.ceil_div(m_size, self.block_m), self.utils.ceil_div(self.n_size, self.block_n)]
        self.attrs.warps = 1

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, self.k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[self.k_size, self.n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.block_k, self.block_n])
        acc = self.register_tensor(dtype=float32, layout=self.mma.lc, init=0.0)

        for offset_k in range(0, self.k_size, self.block_k):
            lda = self.load_global(ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k])
            self.store_shared(sa, lda, offsets=[0, 0])
            ldb = self.load_global(gb, offsets=[offset_k, offset_n], shape=[self.block_k, self.block_n])
            self.store_shared(sb, ldb, offsets=[0, 0])
            self.sync()

            a = self.load_shared(sa, out_layout=self.mma.la)
            b = self.load_shared(sb, out_layout=self.mma.lb)
            acc = self.mma_dot(a, b, acc, mma_inst=self.mma.name, warp_spatial=(1, 1, 1), warp_repeat=(1, 1, 1))
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, self.n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


@pytest.mark.parametrize("m", [129, 257, 511])
@pytest.mark.parametrize("n,k", [[234, 456]])
def test_matmul_v1(m, n, k):
    matmul = MatmulV1(n, k)
    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c = torch.empty(m, n, dtype=torch.float16).cuda()
    c_ref = a @ b

    matmul.kernel(m, a, b, c)

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual=c,
        expected=c_ref,
    )
