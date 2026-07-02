# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Single-MMA FP8 × FP4 starter for the mega-MoE port.

This is the smallest possible correctness exercise for the matmul piece of
mega-MoE: load a 2D FP4 matrix and a 2D FP8 matrix into SMEM, issue ONE
``tcgen05.mma`` call, store the FP32 accumulator back to global memory.

We intentionally start with the *unscaled* ``tcgen05.mma.kind::f8f6f4`` variant
because tilus already supports it (see
``python/tilus/backends/emitters/cuda/tcgen05/mma.py``: any combination of
``{float8_e4m3, float8_e5m2, float6_e2m3, float4_e2m1}`` on either operand
selects ``kind::f8f6f4``).

Once this baseline is solid, we extend to the block-scaled
``kind::mxf8f6f4.block_scale`` variant which mega-MoE actually needs (UE8M0
scale factors per 32-K block). The ``Mma{S}cgen05Example`` skeleton at the
bottom of this file marks the call sites that will change.

Reference math
--------------
For an MMA of shape ``[M, N, K]`` with operands::

    A [M, K]  in dtype_a (e.g. float4_e2m1)
    B [N, K]  in dtype_b (e.g. float8_e4m3)
    D [M, N]  in float32

the kernel computes ``D = A @ B^T`` via one ``tcgen05.mma.kind::f8f6f4``
instruction. (Note ``b`` is fed as ``b.transpose()`` to the MMA so the K axis
of B is contracted.)
"""

from __future__ import annotations

import pytest
import torch

import tilus
import tilus.testing
from tilus import float4_e2m1, float8_e4m3, float32, void_p
from tilus.hidet.ir.type import DataType


# ---------------------------------------------------------------------------
# Helpers for FP4 / FP8 reference values
# ---------------------------------------------------------------------------

def make_fp_operand(
    shape: tuple[int, ...],
    dtype: DataType,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a random FP32 tensor to ``dtype`` and return ``(storage, ref)``.

    - ``storage`` is the packed-byte torch tensor backing the tilus tensor of
      ``dtype``. Pass this to the kernel via ``.data_ptr()``.
    - ``ref`` is an FP32 torch tensor holding the *exact* values that ``dtype``
      represents (i.e., the result of a round-trip ``fp32 -> dtype -> fp32``).
      Use this for the reference matmul.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(shape, dtype=torch.float32, device="cuda", generator=g)

    # Round-trip: fp32 -> dtype -> fp32 to get the representable values, then
    # store as ``dtype`` again to get the packed-byte storage tensor.
    ref = tilus.from_torch(raw).to(dtype).to(float32).torch()
    storage = tilus.from_torch(ref).to(dtype).storage
    return storage, ref


# ---------------------------------------------------------------------------
# Baseline: unscaled tcgen05.mma.kind::f8f6f4   (works today)
# ---------------------------------------------------------------------------

class Tcgen05MmaF8F6F4Example(tilus.Script):
    """Single-MMA mixed-precision matmul: ``D = A @ B^T``.

    A may be any of ``{float8_e4m3, float8_e5m2, float6_e2m3, float4_e2m1}``.
    B independently may be any of the same set. The emitter picks
    ``kind::f8f6f4``. No scale factors.
    """

    def __init__(
        self,
        a_dtype: DataType,
        b_dtype: DataType,
        d_dtype: DataType,
        mma_m: int,
        mma_n: int,
        mma_k: int,
    ):
        super().__init__()
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.d_dtype = d_dtype
        self.mma_m = mma_m
        self.mma_n = mma_n
        self.mma_k = mma_k

    def __call__(self, a_ptr: void_p, b_ptr: void_p, d_ptr: void_p) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=self.a_dtype, shape=[self.mma_m, self.mma_k])
        g_b = self.global_view(b_ptr, dtype=self.b_dtype, shape=[self.mma_n, self.mma_k])
        g_d = self.global_view(d_ptr, dtype=self.d_dtype, shape=[self.mma_m, self.mma_n])

        s_a = self.shared_tensor(dtype=self.a_dtype, shape=[self.mma_m, self.mma_k])
        s_b = self.shared_tensor(dtype=self.b_dtype, shape=[self.mma_n, self.mma_k])
        t_d = self.tcgen05.alloc(dtype=float32, shape=[self.mma_m, self.mma_n])

        mbarriers = self.mbarrier.alloc(counts=[1, 1])
        tma_mbarrier = mbarriers[0]
        mma_mbarrier = mbarriers[1]
        self.sync()

        # Load A and B from global to shared
        with self.single_warp():
            with self.single_thread():
                self.mbarrier.arrive_and_expect_tx(
                    tma_mbarrier, transaction_bytes=s_a.nbytes + s_b.nbytes
                )
            self.tma.global_to_shared(src=g_a, dst=s_a, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_b, dst=s_b, offsets=[0, 0], mbarrier=tma_mbarrier)
        self.mbarrier.wait(tma_mbarrier, phase=0)

        # Issue MMA
        with self.single_warp():
            self.tcgen05.mma(a=s_a, b=s_b.transpose(), d=t_d, enable_input_d=False)
            self.tcgen05.commit(mma_mbarrier)
        self.mbarrier.wait(mma_mbarrier, phase=0)

        # Store accumulator to global
        r_d = self.tcgen05.load(t_d)
        r_d_out = self.cast(r_d, dtype=self.d_dtype)
        self.store_global(g_d, r_d_out, offsets=[0, 0])
        self.tcgen05.dealloc(t_d)


# ---------------------------------------------------------------------------
# Target (NOT YET WORKING): block-scaled tcgen05.mma.kind::mxf8f6f4
# ---------------------------------------------------------------------------
# This is the variant mega-MoE actually uses. It takes UE8M0 scale factors per
# 32-K-element block on both A and B, loaded into TMEM via UTCCP from SMEM.
#
# To enable this in tilus we need to add (Phase 4b in roadmap.md):
#   1. UE8M0 scale-factor dtype.
#   2. SF tensor in TMEM (allocated separately or as a sub-region of t_d's
#      TMEM allocation).
#   3. UTCCP SMEM->TMEM copy primitive.
#   4. ``self.tcgen05.mma`` extended to accept ``sfa=`` and ``sfb=`` args, or
#      a new ``self.tcgen05.mma_block_scaled(...)`` overload.
#
# The skeleton below shows the shape we expect the API to take. It currently
# calls the unscaled MMA so this script still runs end-to-end; lines marked
# with ``# TODO mxf8f6f4`` are the ones that will change once SF support
# lands.

class Tcgen05MmaMxF8F6F4Skeleton(tilus.Script):
    """Block-scaled FP8×FP4 MMA scaffold (target API).

    Shapes::

        A    [M, K]      a_dtype  (e.g. float4_e2m1)
        B    [N, K]      b_dtype  (e.g. float8_e4m3)
        SFA  [M, K/32]   ue8m0    -- one scale per (m_row, K-32-block) of A
        SFB  [N, K/32]   ue8m0    -- one scale per (n_col, K-32-block) of B
        D    [M, N]      float32

    The MMA computes::

        D[m, n] = sum_{kb} sum_{i in 0..31}
                    dequant(A[m, kb*32+i], SFA[m, kb])
                  * dequant(B[n, kb*32+i], SFB[n, kb])
    """

    def __init__(
        self,
        a_dtype: DataType,
        b_dtype: DataType,
        d_dtype: DataType,
        mma_m: int,
        mma_n: int,
        mma_k: int,
    ):
        super().__init__()
        assert mma_k % 32 == 0, "mxf8f6f4 requires K divisible by 32"
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.d_dtype = d_dtype
        self.mma_m = mma_m
        self.mma_n = mma_n
        self.mma_k = mma_k
        self.sf_k = mma_k // 32  # number of K-32 blocks

    def __call__(
        self,
        a_ptr: void_p,
        b_ptr: void_p,
        sfa_ptr: void_p,  # TODO mxf8f6f4: will be ue8m0 storage
        sfb_ptr: void_p,
        d_ptr: void_p,
    ) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=self.a_dtype, shape=[self.mma_m, self.mma_k])
        g_b = self.global_view(b_ptr, dtype=self.b_dtype, shape=[self.mma_n, self.mma_k])
        g_d = self.global_view(d_ptr, dtype=self.d_dtype, shape=[self.mma_m, self.mma_n])
        # TODO mxf8f6f4: g_sfa/g_sfb as ue8m0 [M, sf_k] / [N, sf_k]

        s_a = self.shared_tensor(dtype=self.a_dtype, shape=[self.mma_m, self.mma_k])
        s_b = self.shared_tensor(dtype=self.b_dtype, shape=[self.mma_n, self.mma_k])
        t_d = self.tcgen05.alloc(dtype=float32, shape=[self.mma_m, self.mma_n])
        # TODO mxf8f6f4:
        #   s_sfa = self.shared_tensor(dtype=ue8m0, shape=[mma_m, sf_k])
        #   s_sfb = self.shared_tensor(dtype=ue8m0, shape=[mma_n, sf_k])
        #   t_sfa = self.tcgen05.alloc_sf(...)   # via UTCCP
        #   t_sfb = self.tcgen05.alloc_sf(...)

        mbarriers = self.mbarrier.alloc(counts=[1, 1])
        tma_mbarrier = mbarriers[0]
        mma_mbarrier = mbarriers[1]
        self.sync()

        with self.single_warp():
            with self.single_thread():
                self.mbarrier.arrive_and_expect_tx(
                    tma_mbarrier,
                    transaction_bytes=s_a.nbytes + s_b.nbytes,  # TODO mxf8f6f4: + s_sfa.nbytes + s_sfb.nbytes
                )
            self.tma.global_to_shared(src=g_a, dst=s_a, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_b, dst=s_b, offsets=[0, 0], mbarrier=tma_mbarrier)
            # TODO mxf8f6f4: tma load s_sfa, s_sfb
        self.mbarrier.wait(tma_mbarrier, phase=0)

        with self.single_warp():
            # TODO mxf8f6f4:
            #   self.tcgen05.utccp(src=s_sfa, dst=t_sfa)
            #   self.tcgen05.utccp(src=s_sfb, dst=t_sfb)
            #   self.tcgen05.mma(a=s_a, b=s_b.transpose(), d=t_d,
            #                    sfa=t_sfa, sfb=t_sfb,
            #                    enable_input_d=False)
            self.tcgen05.mma(a=s_a, b=s_b.transpose(), d=t_d, enable_input_d=False)
            self.tcgen05.commit(mma_mbarrier)
        self.mbarrier.wait(mma_mbarrier, phase=0)

        r_d = self.tcgen05.load(t_d)
        r_d_out = self.cast(r_d, dtype=self.d_dtype)
        self.store_global(g_d, r_d_out, offsets=[0, 0])
        self.tcgen05.dealloc(t_d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@tilus.testing.requires.nvgpu_sm100a
@pytest.mark.parametrize(
    "a_dtype, b_dtype, mma_m, mma_n, mma_k",
    [
        # Same-dtype baselines (sanity checks)
        (float8_e4m3, float8_e4m3, 128, 32, 32),
        (float4_e2m1, float4_e2m1, 128, 32, 32),
        # The target combination for mega-MoE: A=FP4 (weight), B=FP8 (acts)
        (float4_e2m1, float8_e4m3, 128, 32, 32),
        (float4_e2m1, float8_e4m3, 128, 16, 32),
        (float4_e2m1, float8_e4m3, 128, 64, 32),
        # Reverse swap, just to confirm independence of A and B dtype fields
        (float8_e4m3, float4_e2m1, 128, 32, 32),
    ],
)
def test_mma_f8f6f4(a_dtype, b_dtype, mma_m, mma_n, mma_k):
    a_storage, a_ref = make_fp_operand((mma_m, mma_k), a_dtype, seed=0)
    b_storage, b_ref = make_fp_operand((mma_n, mma_k), b_dtype, seed=1)
    d = torch.empty(mma_m, mma_n, dtype=torch.float32, device="cuda")

    kernel = Tcgen05MmaF8F6F4Example(a_dtype, b_dtype, float32, mma_m, mma_n, mma_k)
    kernel(a_storage, b_storage, d)
    torch.cuda.synchronize()

    expected = a_ref.to(torch.float32) @ b_ref.to(torch.float32).T
    torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
