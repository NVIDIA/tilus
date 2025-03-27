import math

import pytest
import tilus
import torch
from tilus import float16, float32, int32


class MatmulV0(tilus.Script):
    def __init__(self):
        super().__init__()
        self.mma = self.cuda.mma_configs.m16n8k16_f16_f32

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
        acc = self.register_tensor(dtype=float32, layout=self.mma.lc, f_init=lambda indices: float32.zero)

        k_blocks = self.utils.ceil_div(k_size, self.block_k)
        for k in range(k_blocks):
            offset_k = k * self.block_k

            a = self.load_global(ga, offsets=[offset_m, offset_k], dims=[0, 1], layout=self.mma.la)
            b = self.load_global(gb, offsets=[offset_k, offset_n], dims=[0, 1], layout=self.mma.lb)
            acc = self.mma_dot(a, b, acc, config=self.mma, warp_spatial=(1, 1, 1), warp_repeat=(1, 1, 1))

        acc_f16 = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, acc_f16, offsets=[offset_m, offset_n])


@pytest.mark.parametrize("m", [333, 444, 555])
@pytest.mark.parametrize("n, k", [[333, 444]])
def test_kernels_matmul_v0(m, n, k):
    matmul = MatmulV0()
    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c = torch.empty(m, n, dtype=torch.float16).cuda()
    c_ref = a @ b

    matmul(m, n, k, a, b, c)

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual=c,
        expected=c_ref,
    )
