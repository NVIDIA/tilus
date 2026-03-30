"""
Test R16x256B tcgen05 load with increasing complexity to find where it breaks:
1. Single CTA, 4 warps, no MMA (simple round-trip) - known working
2. Single CTA, 4 warps, with tcgen05 MMA
3. Two CTAs (cluster), with tcgen05 MMA (matmul_v9 style)
"""

import os
import shutil
import torch
import tilus
from tilus import float16, float32, int32

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_multicta")


class SingleCtaMma(tilus.Script):
    """Single CTA: tcgen05 MMA -> tcgen05.load -> cast -> store_shared -> TMA out"""

    debug_schedule = dict(block_m=128, block_n=32, block_k=64)

    def __init__(self, block_m: int, block_n: int, block_k: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = 1, 1
        self.attrs.warps = 4

        block_m = self.block_m
        block_n = self.block_n
        block_k = self.block_k

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])

        # Load A, B via TMA
        s_a = self.shared_tensor(dtype=float16, shape=[block_m, block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[block_n, block_k])

        with self.single_thread():
            self.tma.global_to_shared(g_a, s_a, offsets=[0, 0])
            self.tma.global_to_shared(g_b, s_b, offsets=[0, 0])
            self.tma.commit_group()
            self.tma.wait_group(n=0)
        self.sync()

        # MMA: cta_group=1 (single CTA)
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[block_m, block_n], cta_group=1)

        with self.single_thread():
            self.tcgen05.mma(s_a, s_b.transpose(), t_acc, enable_input_d=False, cta_group=1)
            self.tcgen05.commit(cta_group=1)

        self.tcgen05.wait_load()

        # Epilogue: tcgen05.load -> cast -> store_shared -> TMA out
        r_acc = self.tcgen05.load(t_acc)
        self.tcgen05.wait_load()
        r_fp16 = r_acc.to(float16)

        s_c = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        self.store_shared(s_c, r_fp16)
        self.fence.async_view(space="shared")
        self.sync()
        with self.single_thread():
            self.tma.shared_to_global(s_c, g_c, offsets=[0, 0], dims=[0, 1])
            self.tma.commit_group()
            self.tma.wait_group(n=0)

        self.sync()
        self.tcgen05.dealloc(t_acc)


def test_single_cta_mma():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    tilus.option.cache_dir(CACHE_DIR)
    tilus.option.debug.dump_ir()

    kernel = SingleCtaMma()

    m, n, k = 128, 32, 64
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c_actual = torch.empty(m, n, dtype=torch.float16, device="cuda")
    c_expected = a @ b.T

    kernel(m, n, k, a, b, c_actual)
    torch.cuda.synchronize()

    if torch.allclose(c_actual, c_expected, atol=1e-1, rtol=1e-1):
        print("Single CTA MMA: PASSED")
    else:
        mismatch = (~torch.isclose(c_actual, c_expected, atol=1e-1, rtol=1e-1)).sum().item()
        total = c_actual.numel()
        print(f"Single CTA MMA: FAILED ({mismatch}/{total} = {mismatch/total*100:.1f}%)")
        diff = (c_actual - c_expected).abs()
        print(f"  Max abs diff: {diff.max().item():.4f}")


if __name__ == "__main__":
    test_single_cta_mma()
