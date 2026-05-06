# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NVFP4 × NVFP4 → BF16 matmul, tutorial version (v0).

Mirrors the structure of ``examples/blackwell_matmul/matmul_v1.py`` (FP16
tutorial with TMA + mbarrier + tcgen05.mma) and adds the NVFP4-specific
pieces:

  - FP4 (E2M1) operands packed 2-per-byte in HBM and SMEM.
  - UE4M3 (= ``float8_e4m3fn``) scale-factor tensors, one scale per 16
    K-elements per M-row of A (and per N-col of B). The SF tensors are
    **pre-shuffled in HBM** into the canonical block-scaled layout via
    ``tilus.testing.shuffle_sf_to_block_scaled_layout`` so that TMA can
    load them directly into the SMEM tile that ``tcgen05.cp`` expects.
  - SF SMEM staging tile of shape ``[K_outer, 32, MN_fold, 4]`` per main-K
    iter — one 512-byte atom per inst-K iter.
  - Per inst-K iter: ``tcgen05.copy(..., multicast="warpx4")`` then
    ``tcgen05.mma_scaled(...)``. Each ``mma_scaled`` call lowers to a
    single PTX ``tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X``
    instruction.

TMEM convention used throughout
-------------------------------
``tcgen05.alloc`` allocates a TMEM tensor of rank ``>= 2``:

* **Dim 0** is the **lane** axis (TMEM rows). Its size is the *unique* lane
  count: 32, 64, or 128.
* **All other dims** are column-strided in element units of the tensor's
  dtype.

For block-scaled MMA, the SF tensor in TMEM has shape
``[32, K_outer, MN_fold, 4]`` per CTA:

* ``[0] = 32`` lanes → ``WARPX4`` duplication (replicated across all 4
  warp sub-partitions, set by ``tcgen05.copy(..., multicast="warpx4")``).
* ``[1] = K_outer`` = ``block_k / inst_K`` → number of inst-K iters per
  main-K loop iter (each iter consumes one 32×MN_fold×4-byte atom).
* ``[2] = MN_fold`` = ``block_m // 32`` (SFA) or ``block_n // 32`` (SFB).
* ``[3] = 4`` → the 4 byte slots in each 32-bit TMEM cell. For
  ``scale_vec::4X`` (NVFP4) all 4 are the K_inner SFs of one inst.
"""

import time

import pandas
import torch

import tilus
from tilus import bfloat16, float4_e2m1, float8_e4m3, float32, int32, uint32
from tilus.testing import NVFP4_SF_BLOCK_K as SF_BLOCK_K
from tilus.testing import (
    dequantize_nvfp4,
    quantize_nvfp4,
    shuffle_sf_to_block_scaled_layout,
)
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")


# NVFP4 with `scale_vec::4X / block16` has fixed inst dimensions:
#   inst_M = 128 (per CTA, cta_group::1)
#   inst_K = 64
INST_M = 128
INST_K = 64


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

# block_m must equal INST_M = 128 (one TMEM-CTA's full M extent for cta_group::1).
# block_n must be a multiple of 32 (PTX inst N step; we use 128/256 for cleanness).
# block_k must be a multiple of INST_K = 64; the SF main-K loop steps in
# `K_outer = block_k / INST_K` inst-K iters per main-K iter.
@tilus.autotune("block_m, block_n", [[128, 128], [128, 256]])
@tilus.autotune("block_k", [128, 256, 512])
class NVFP4MatmulV0(tilus.Script):
    """One-stage NVFP4 matmul: ``D = A @ B^T`` with per-16-K-block UE4M3 scales.

    The SF inputs ``sfa`` and ``sfb`` are in **canonical block-scaled HBM
    layout** (= post-``shuffle_sf_to_block_scaled_layout``). The driver below
    handles that pre-shuffle.
    """

    def __init__(self, block_m: int, block_n: int, block_k: int):
        super().__init__()
        assert block_m == INST_M, f"block_m must be {INST_M} (cta_group::1 inst_M)"
        assert block_n % 32 == 0, "block_n must be a multiple of 32"
        assert block_k % INST_K == 0, f"block_k must be a multiple of inst_K={INST_K}"
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        # Number of inst-K iters per main-K loop iter (= K_outer atoms in HBM/SMEM/TMEM SF tiles).
        self.k_outer = block_k // INST_K

    def __call__(
        self,
        m_size: int32,
        n_size: int32,
        k_size: int,
        a_ptr: ~float4_e2m1,
        b_ptr: ~float4_e2m1,
        sfa_ptr: ~float8_e4m3,  # canonical block-scaled HBM layout (pre-shuffled)
        sfb_ptr: ~float8_e4m3,  # canonical block-scaled HBM layout (pre-shuffled)
        c_ptr: ~bfloat16,
    ):
        # One CTA per (M-tile, N-tile). 1 warpgroup (4 warps).
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)]
        self.attrs.warps = 4

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        k_outer = self.k_outer
        m_fold = block_m // 32  # SFA M-fold count = 4 for block_m=128
        n_fold = block_n // 32  # SFB N-fold count
        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        # ── Global views ───────────────────────────────────────────────────
        # A and B are FP4 logical [M, K] / [N, K]; storage is packed 2-per-byte.
        g_a = self.global_view(a_ptr, dtype=float4_e2m1, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float4_e2m1, shape=[n_size, k_size])
        # SF in HBM — pre-shuffled; the (M, K_blocks) view is preserved (same
        # byte count) but the byte order is the canonical block-scaled layout
        # so a contiguous (block_m, block_k // SF_BLOCK_K) box load lands in
        # SMEM in the order tcgen05.cp.32x128b.warpx4 wants.
        g_sfa = self.global_view(sfa_ptr, dtype=float8_e4m3, shape=[m_size, k_size // SF_BLOCK_K])
        g_sfb = self.global_view(sfb_ptr, dtype=float8_e4m3, shape=[n_size, k_size // SF_BLOCK_K])

        # ── SMEM tiles ─────────────────────────────────────────────────────
        s_a = self.shared_tensor(dtype=float4_e2m1, shape=[block_m, block_k])
        s_b = self.shared_tensor(dtype=float4_e2m1, shape=[block_n, block_k])
        # SF SMEM tile: declared 2-D `(block_rows, block_k/16)` so TMA can box-
        # load the canonical-layout HBM SF tensor directly (no SMEM-side
        # shuffle). For tcgen05.cp we re-interpret it as 3-D
        # `(k_outer, 32, MN_fold*4)` via reshape_shared — same 1024- (or
        # 2048-, etc.) byte sequence, different indexing.
        s_sfa = self.shared_tensor(dtype=float8_e4m3, shape=[block_m, block_k // SF_BLOCK_K])
        s_sfb = self.shared_tensor(dtype=float8_e4m3, shape=[block_n, block_k // SF_BLOCK_K])

        # ── TMEM allocations ───────────────────────────────────────────────
        # FP32 accumulator: one TMEM cell per output element. shape[0] = 128
        # → duplication = NONE (all 128 lanes unique).
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[block_m, block_n])
        # SF tensors in TMEM: [32 lanes, K_outer, MN_fold, 4 K_inner]. The
        # K_outer dim holds one 32×MN_fold×4-byte atom per inst-K iter.
        # shape[0] = 32 → duplication = WARPX4 (the SFs are duplicated across
        # all 4 warp sub-partitions per PTX 9.7.16.10.7).
        t_sfa = self.tcgen05.alloc(dtype=float8_e4m3, shape=[32, k_outer, m_fold, 4])
        t_sfb = self.tcgen05.alloc(dtype=float8_e4m3, shape=[32, k_outer, n_fold, 4])

        # Three mbarriers: TMA completion + SF copy completion + MMA completion.
        sf_barrier, tma_barrier, mma_barrier = self.mbarrier.alloc(counts=[1, 1, 1]).tolist()
        phase: uint32 = 0

        self.sync()

        # ── Main K loop ────────────────────────────────────────────────────
        for offset_k in range(0, k_size, self.block_k):
            with self.single_warp():
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(
                        tma_barrier,
                        transaction_bytes=(
                            s_a.nbytes + s_b.nbytes + s_sfa.nbytes + s_sfb.nbytes
                        ),
                    )

                # TMA loads: A, B, and the two SF tiles in parallel. The SF
                # box loads (block_m × block_k/16) bytes from canonical-layout
                # HBM into the SMEM tile, where the byte order is already
                # (K_outer, lane, MN_fold, K_inner) — direct match for the
                # tcgen05.cp source descriptor's atom layout.
                self.tma.global_to_shared(src=g_a, dst=s_a, offsets=[offset_m, offset_k], mbarrier=tma_barrier)
                self.tma.global_to_shared(src=g_b, dst=s_b, offsets=[offset_n, offset_k], mbarrier=tma_barrier)
                self.tma.global_to_shared(
                    src=g_sfa, dst=s_sfa, offsets=[offset_m, offset_k // SF_BLOCK_K], mbarrier=tma_barrier
                )
                self.tma.global_to_shared(
                    src=g_sfb, dst=s_sfb, offsets=[offset_n, offset_k // SF_BLOCK_K], mbarrier=tma_barrier
                )
                self.mbarrier.wait(tma_barrier, phase=phase)

                # SMEM → TMEM SF copy: one tcgen05.cp.32x128b.warpx4 per
                # K_outer iter. Each writes 32 lanes × 16 bytes per lane = 512
                # bytes (= one inst's SFA worth) into the matching K_outer slot
                # of t_sf{a,b}. We reshape the 2-D SMEM tile to 3-D
                # `(k_outer, 32, MN_fold*4)` so each per-inst atom is the
                # natural 2-D `(32, MN_fold*4)` slice the cp wants.
                s_sfa_atoms = self.reshape_shared(s_sfa, [k_outer, 32, m_fold * 4])
                s_sfb_atoms = self.reshape_shared(s_sfb, [k_outer, 32, n_fold * 4])
                for k in range(k_outer):
                    self.tcgen05.copy(src=s_sfa_atoms[k], dst=t_sfa[:, k, :, :], multicast="warpx4")
                    self.tcgen05.copy(src=s_sfb_atoms[k], dst=t_sfb[:, k, :, :], multicast="warpx4")
                self.tcgen05.commit(mbarrier=sf_barrier)
                self.mbarrier.wait(sf_barrier, phase=phase)

                # Block-scaled MMA: one mma_scaled call per inst-K iter. Each
                # lowers to a single PTX
                #   tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.block16
                # (NVFP4 with UE4M3 SFs ⇒ scale_vec::4X.block16 inferred from
                # operand dtypes + sf_block_size).
                #
                # To pick per-inst K-slabs we reshape s_a / s_b to
                # (block_m_or_n, k_outer, INST_K) and index with `[:, k, :]`.
                # (Python `[a:b]` slicing on tilus tensors is intentionally
                # not supported — see CLAUDE / tilus convention.)
                s_a_chunks = self.reshape_shared(s_a, [block_m, k_outer, INST_K])
                s_b_chunks = self.reshape_shared(s_b, [block_n, k_outer, INST_K])
                for k in range(k_outer):
                    # For k > 0 we always accumulate (within this main-K iter we already
                    # produced a partial D). For k == 0 we only accumulate if it's not the
                    # first main-K iter (offset_k != 0). The branch on `k` is a Python int
                    # so it's resolved at transpile time.
                    enable_input_d = True if k != 0 else (offset_k != 0)
                    self.tcgen05.mma_scaled(
                        a=s_a_chunks[:, k, :],                  # (block_m, INST_K)
                        b=s_b_chunks[:, k, :].transpose(),      # (INST_K, block_n)
                        d=t_acc,                                # (block_m, block_n)
                        sfa=t_sfa[:, k, :, :],                  # [32, m_fold, 4]
                        sfb=t_sfb[:, k, :, :],                  # [32, n_fold, 4]
                        sf_block_size=16,
                        enable_input_d=enable_input_d,
                    )
                self.tcgen05.commit(mbarrier=mma_barrier)
                self.mbarrier.wait(mma_barrier, phase=phase)

            self.sync()
            phase ^= 1

        # ── Epilogue: TMEM → registers → global (direct) ───────────────────
        r_acc = self.tcgen05.load(t_acc)
        g_c = self.global_view(c_ptr, dtype=bfloat16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(bfloat16), offsets=[offset_m, offset_n])

        # All TMEM allocations must be released before the kernel exits.
        self.sync()
        self.tcgen05.dealloc(t_sfa)
        self.tcgen05.dealloc(t_sfb)
        self.tcgen05.dealloc(t_acc)


# ---------------------------------------------------------------------------
# Driver: run, validate, benchmark
# ---------------------------------------------------------------------------

def main(bench: bool = True):
    matmul = NVFP4MatmulV0()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows = []

    for m_size, n_size, k_size in [
        [4096, 4096, 4096],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")

        # Generate FP32 reference inputs and quantize to NVFP4.
        torch.manual_seed(0)
        a_fp32 = torch.randn(m_size, k_size, dtype=torch.float32, device="cuda")
        b_fp32 = torch.randn(n_size, k_size, dtype=torch.float32, device="cuda")

        a_packed, sfa_natural = quantize_nvfp4(a_fp32)
        b_packed, sfb_natural = quantize_nvfp4(b_fp32)

        a_dequant = dequantize_nvfp4(a_packed, sfa_natural)
        b_dequant = dequantize_nvfp4(b_packed, sfb_natural)
        c_ref = (a_dequant @ b_dequant.T).to(torch.bfloat16)

        # Pre-shuffle SFs into the canonical block-scaled HBM layout.
        sfa = shuffle_sf_to_block_scaled_layout(sfa_natural, sf_block_size=16).contiguous()
        sfb = shuffle_sf_to_block_scaled_layout(sfb_natural, sf_block_size=16).contiguous()

        # Output buffer.
        c = torch.empty(m_size, n_size, dtype=torch.bfloat16, device="cuda")

        matmul(m_size, n_size, k_size, a_packed, b_packed, sfa, sfb, c)
        torch.cuda.synchronize()

        torch.testing.assert_close(c, c_ref, atol=2e-2, rtol=2e-2)

        # cuBLASLt baseline via torch._scaled_mm. Note: torch._scaled_mm
        # requires the natural-layout SFs (it does its own pre-shuffle internally).
        a_view = a_packed.view(torch.float4_e2m1fn_x2)
        b_view = b_packed.view(torch.float4_e2m1fn_x2)

        def torch_baseline():
            return torch._scaled_mm(
                a_view, b_view.t(),
                scale_a=sfa_natural, scale_b=sfb_natural.t(),
                out_dtype=torch.bfloat16,
            )

        if bench:
            for name, func in [
                ("torch (cublasLt)", torch_baseline),
                ("tilus", lambda: matmul(
                    m_size, n_size, k_size, a_packed, b_packed, sfa, sfb, c)),
            ]:
                latency = benchmark_func(func, warmup=5, repeat=100)
                tflops = 2 * m_size * n_size * k_size / latency * 1e-9
                rows.append([m_size, n_size, k_size, name, latency, tflops])
                time.sleep(3)  # cool down between runs

    if bench:
        df = pandas.DataFrame(rows, columns=headers)
        print(df)


if __name__ == "__main__":
    main(bench=True)
