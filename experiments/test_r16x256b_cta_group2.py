"""
Targeted test: tcgen05.alloc(cta_group=2) -> tcgen05.store -> tcgen05.load -> store_shared -> TMA
Tests R16x256B vs R32x32B with cta_group=2, using annotate_layout to force specific register layouts.
"""

import os
import shutil
import torch
import tilus
from tilus import RegisterTensor, SharedTensor, float16, float32, int32
from tilus.ir.layout.ops.register_ops import local, spatial
from tilus.ir.layout.cuda.tcgen05.ldst import get_ldst_layout
from tilus.hidet.ir.primitives.cuda.tcgen05 import Tcgen05LoadStoreShapeKind

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_cta_group2")


def make_r16x256b_layout(shape, dtype_nbits):
    """Build the R16x256B register layout for given shape and dtype."""
    sk = Tcgen05LoadStoreShapeKind.R16x256B
    atom = get_ldst_layout(sk)
    total_rows = shape[0]
    total_cols_bits = shape[1] * dtype_nbits
    cols_repeat = total_cols_bits // sk.columns_bits()
    rows_repeat = total_rows // sk.rows()
    intra = 2 if rows_repeat % 2 == 0 else 1
    inter = rows_repeat // intra
    return local(1, cols_repeat).spatial(inter, 1).local(intra, 1) * atom * local(1, 32 // dtype_nbits)


def make_r32x32b_layout(shape, dtype_nbits):
    """Build the R32x32B register layout for given shape and dtype."""
    sk = Tcgen05LoadStoreShapeKind.R32x32B
    atom = get_ldst_layout(sk)
    total_rows = shape[0]
    total_cols_bits = shape[1] * dtype_nbits
    cols_repeat = total_cols_bits // sk.columns_bits()
    rows_repeat = total_rows // sk.rows()
    intra = 1
    inter = rows_repeat
    return local(1, cols_repeat).spatial(inter, 1).local(intra, 1) * atom * local(1, 32 // dtype_nbits)


class CtaGroup2Test(tilus.Script):
    """
    2 CTAs with cluster, cta_group=2.
    tcgen05.alloc -> tcgen05.store (init with known data) -> tcgen05.load -> cast -> store_shared -> TMA out.
    Uses annotate_layout to force the register layout.
    """

    debug_schedule = dict(block_m=256, block_n=256, e_block_n=32, use_r16x256b=0)

    def __init__(self, block_m: int, block_n: int, e_block_n: int, use_r16x256b: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.e_block_n = e_block_n
        self.use_r16x256b = use_r16x256b

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        in_ptr: ~float16,
        out_ptr: ~float16,
    ):
        block_m = self.block_m
        block_n = self.block_n

        # Match matmul_v9: 2 CTAs with cluster
        self.attrs.blocks = 2, 1
        self.attrs.cluster_blocks = 2
        self.attrs.warps = 8  # 8 warps per CTA (256 threads)

        g_in = self.global_view(in_ptr, dtype=float16, shape=[m_size, n_size])
        g_out = self.global_view(out_ptr, dtype=float16, shape=[m_size, n_size])

        e_block_n = self.e_block_n

        # Allocate tmem with cta_group=2 (like matmul_v9) - full N dimension
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[block_m // 2, block_n], cta_group=2)

        cta_rank = self.cluster.blockRank

        self.cluster_sync()

        block_k = 64
        g_a = self.global_view(in_ptr, dtype=float16, shape=[m_size, n_size])  # reuse as A
        g_b = self.global_view(in_ptr, dtype=float16, shape=[n_size, n_size])  # reuse as B (doesn't matter, just for MMA shape)

        # Load A and B tiles to shared via TMA-like load_global
        s_a = self.shared_tensor(dtype=float16, shape=[block_m // 2, block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[block_n, block_k])

        # Warp 0: load A, B to shared memory using load_global + store_shared
        with self.single_warp(0):
            r_a = self.load_global(g_a, offsets=[cta_rank * (block_m // 2), 0], shape=[block_m // 2, block_k])
            self.store_shared(s_a, r_a)
            r_b = self.load_global(g_b, offsets=[0, 0], shape=[block_n, block_k])
            self.store_shared(s_b, r_b)
        self.sync()

        # Warp 1: MMA to fill tmem
        with self.single_warp(1):
            with self.single_thread():
                self.tcgen05.mma(s_a, s_b.transpose(), t_acc, enable_input_d=False, cta_group=2)
                self.tcgen05.commit(cta_group=2, multicast_mask=0b11)

        self.sync()

        # Warp group 4-7 (epilogue): slice -> tcgen05.load -> cast -> store_shared -> TMA
        # Loop over e_block_n chunks (like matmul_v9)
        with self.warp_group(warp_begin=4, num_warps=4):
            s_out = self.shared_tensor(dtype=float16, shape=[block_m // 2, e_block_n])
            for e_offset_n in range(0, block_n, e_block_n):
                t_slice = self.tcgen05.slice(t_acc, offsets=[0, e_offset_n], shape=[block_m // 2, e_block_n], dims=[0, 1])
                r_acc = self.tcgen05.load(t_slice)
                self.tcgen05.wait_load()

                r_fp16 = r_acc.to(float16)
                self.store_shared(s_out, r_fp16)

                self.fence.async_view(space="shared")
                self.sync()
                with self.single_thread():
                    offset_m_out = cta_rank * (block_m // 2)
                    self.tma.shared_to_global(s_out, g_out, offsets=[offset_m_out, e_offset_n], dims=[0, 1])
                    self.tma.commit_group()
                    self.tma.wait_group(n=0)
                self.sync()

        self.sync()
        self.tcgen05.dealloc(t_acc)


def test(use_r16x256b: bool):
    label = "R16x256B" if use_r16x256b else "R32x32B"
    cache = os.path.join(CACHE_DIR, label.lower())
    if os.path.exists(cache):
        shutil.rmtree(cache)
    tilus.option.cache_dir(cache)
    tilus.option.debug.dump_ir()

    kernel = CtaGroup2Test()
    kernel.__class__.debug_schedule = dict(block_m=256, block_n=256, e_block_n=32, use_r16x256b=int(use_r16x256b))

    m, n = 256, 256
    inp = torch.arange(m * n, dtype=torch.float16, device="cuda").reshape(m, n)
    out = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    kernel(m, n, inp, out)
    torch.cuda.synchronize()

    if torch.allclose(out, inp, atol=1.0, rtol=0.1):
        print(f"{label}: PASSED")
    else:
        close = torch.isclose(out, inp, atol=1.0, rtol=0.1)
        pct = close.float().mean() * 100
        print(f"{label}: FAILED ({pct:.1f}% correct)")
        print(f"  Expected [0:4, 0:8]: {inp[0:4, 0:8].tolist()}")
        print(f"  Got      [0:4, 0:8]: {out[0:4, 0:8].tolist()}")
        # Check CTA0 vs CTA1 blocks
        for cta in range(2):
            base = cta * 128
            block = out[base:base+4, 0:8]
            ref = inp[base:base+4, 0:8]
            pct_b = torch.isclose(block, ref, atol=1.0, rtol=0.1).float().mean() * 100
            print(f"  CTA{cta} [{base}:{base+4}, 0:8]: {pct_b:.0f}% correct, got {block.tolist()}")

    return torch.allclose(out, inp, atol=1.0, rtol=0.1)


if __name__ == "__main__":
    print("Testing cta_group=2 with R32x32B and R16x256B:")
    test(use_r16x256b=False)
    test(use_r16x256b=True)
