# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Blackwell NVFP4 GEMM example using tcgen05.scaled_mma.

This example demonstrates the use of nvfp4 (float4_e2m1) tensor core MMA with
block-scale factors (UE8M0 format). The computation performed is:

    D = (A * scale_A) @ (B * scale_B)

where:
  - A is an (M, K) matrix of float4_e2m1 values
  - B is a (K, N) matrix of float4_e2m1 values (stored as (N, K) row-major)
  - scale_A and scale_B are UE8M0 scale factors, one per block of 32 K-elements
  - D is the (M, N) output matrix in float16

This uses the tcgen05 .kind::mxf4 with .block_scale mode on Blackwell (sm_100a).

Usage:
    python nvfp4_matmul.py
"""

import tilus
import torch
from tilus import float4_e2m1, float16, float32, uint8, void_p

# Constants
BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 64

# E2M1 value 1.0 = 0b0010 = 0x2
FP4_ONE = 0x2

# UE8M0 value 1.0 = 2^(127-127) = 2^0 = 0x7F
UE8M0_ONE = 0x7F


class NvFP4Matmul(tilus.Script):
    """Single-tile nvfp4 matmul using tcgen05.scaled_mma.

    This is a minimal example that computes one tile of the output matrix D.
    A real GEMM kernel would tile over M, N, and K with pipelining.
    """

    def __init__(self, block_m: int = BLOCK_M, block_n: int = BLOCK_N, block_k: int = BLOCK_K):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def __call__(
        self,
        a_ptr: void_p,
        b_ptr: void_p,
        scale_a_ptr: void_p,
        scale_b_ptr: void_p,
        d_ptr: void_p,
    ) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 4

        # -- Global memory views --
        # A: (M, K) fp4, B: (N, K) fp4 (row-major, transposed for MMA)
        g_a = self.global_view(a_ptr, dtype=float4_e2m1, shape=[self.block_m, self.block_k])
        g_b = self.global_view(b_ptr, dtype=float4_e2m1, shape=[self.block_n, self.block_k])

        # Scale factors: padded to (128, 128) uint8 for tcgen05.copy compatibility.
        # Actual data occupies the first K/32 bytes per row, first M/N rows respectively.
        g_scale_a = self.global_view(scale_a_ptr, dtype=uint8, shape=[128, 128])
        g_scale_b = self.global_view(scale_b_ptr, dtype=uint8, shape=[128, 128])

        # Output: (M, N) float16
        g_d = self.global_view(d_ptr, dtype=float16, shape=[self.block_m, self.block_n])

        # -- Shared memory --
        s_a = self.shared_tensor(dtype=float4_e2m1, shape=[self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float4_e2m1, shape=[self.block_n, self.block_k])
        s_scale_a = self.shared_tensor(dtype=uint8, shape=[128, 128])
        s_scale_b = self.shared_tensor(dtype=uint8, shape=[128, 128])

        # -- Tensor memory --
        t_d = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n])
        t_scale_a = self.tcgen05.alloc(dtype=uint8, shape=[128, 128])
        t_scale_b = self.tcgen05.alloc(dtype=uint8, shape=[128, 128])

        # Barriers
        mbarriers = self.mbarrier.alloc(counts=[1, 1, 1])
        tma_mbarrier = mbarriers[0]
        copy_mbarrier = mbarriers[1]
        mma_mbarrier = mbarriers[2]
        self.sync()

        # 1. TMA: global → shared for fp4 data and scale factors
        with self.single_thread():
            total_bytes = s_a.nbytes + s_b.nbytes + s_scale_a.nbytes + s_scale_b.nbytes
            self.mbarrier.arrive_and_expect_tx(tma_mbarrier, transaction_bytes=total_bytes)
            self.tma.global_to_shared(src=g_a, dst=s_a, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_b, dst=s_b, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_scale_a, dst=s_scale_a, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_scale_b, dst=s_scale_b, offsets=[0, 0], mbarrier=tma_mbarrier)
        self.mbarrier.wait(tma_mbarrier, phase=0)

        # 2. tcgen05.copy: shared → tensor memory for scale factors
        with self.single_thread():
            self.tcgen05.copy(src=s_scale_a, dst=t_scale_a)
            self.tcgen05.copy(src=s_scale_b, dst=t_scale_b)
            self.tcgen05.commit(copy_mbarrier)
        self.mbarrier.wait(copy_mbarrier, phase=0)

        # 3. Block-scaled MMA: D = (A * scaleA) @ (B^T * scaleB)
        with self.single_thread():
            self.tcgen05.scaled_mma(
                a=s_a,
                b=s_b.transpose(),
                d=t_d,
                scale_a=t_scale_a,
                scale_b=t_scale_b,
                enable_input_d=False,
            )
            self.tcgen05.commit(mma_mbarrier)
        self.mbarrier.wait(mma_mbarrier, phase=0)

        # 4. Epilogue: tensor memory → registers → global
        r_d = self.tcgen05.load(t_d)
        self.tcgen05.wait_load()
        r_d_f16 = self.cast(r_d, dtype=float16)
        self.store_global(g_d, r_d_f16, offsets=[0, 0])

        # 5. Free tensor memory
        self.tcgen05.dealloc(t_d)
        self.tcgen05.dealloc(t_scale_a)
        self.tcgen05.dealloc(t_scale_b)


def main():
    M, N, K = BLOCK_M, BLOCK_N, BLOCK_K
    scale_cols = K // 32  # number of UE8M0 scale values per row

    # Prepare fp4 input data (all 1.0, packed 2 per byte)
    fp4_packed_byte = (FP4_ONE << 4) | FP4_ONE  # 0x22
    a = torch.full((M, K // 2), fp4_packed_byte, dtype=torch.uint8, device="cuda")
    b = torch.full((N, K // 2), fp4_packed_byte, dtype=torch.uint8, device="cuda")

    # Prepare scale factors: (128, 128) uint8, padded for tcgen05.copy
    # First `scale_cols` bytes per row contain UE8M0 scale data
    scale_a = torch.zeros(128, 128, dtype=torch.uint8, device="cuda")
    scale_a[:M, :scale_cols] = UE8M0_ONE

    scale_b = torch.zeros(128, 128, dtype=torch.uint8, device="cuda")
    scale_b[:N, :scale_cols] = UE8M0_ONE

    # Output
    d = torch.empty(M, N, dtype=torch.float16, device="cuda")

    # Run kernel
    kernel = NvFP4Matmul(M, N, K)
    kernel(a, b, scale_a, scale_b, d)
    torch.cuda.synchronize()

    # Verify: D[i,j] = sum_k(1.0 * 1.0 * 1.0 * 1.0) = K = 64
    expected = float(K)
    actual_mean = d.float().mean().item()
    print(f"NVFP4 GEMM result: shape={tuple(d.shape)}, mean={actual_mean:.1f} (expected={expected:.1f})")

    if abs(actual_mean - expected) < 1.0:
        print("PASSED!")
    else:
        print(f"FAILED: expected mean={expected}, got {actual_mean}")
        print(f"Sample values: {d[0, :8]}")


if __name__ == "__main__":
    main()
