# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import tilus
import tilus.testing
import torch
from tilus import float4_e2m1, float16, float32, uint8, void_p

# E2M1 encoding for 1.0: sign=0, exp=01, mantissa=0 → 0b0010 = 0x2
# Two 1.0 values packed per byte: 0x22
FP4_ONE = 0x2
FP4_ONE_PACKED = (FP4_ONE << 4) | FP4_ONE  # 0x22

# UE8M0 encoding for scale=1.0: 2^(e-127) = 1.0 → e = 127 = 0x7F
UE8M0_ONE = 0x7F


class Tcgen05ScaledMmaExample(tilus.Script):
    """Test kernel for nvfp4 (E2M1) block-scaled MMA using tcgen05.scaled_mma.

    Computes: D = (A * scaleA) @ (B * scaleB)
    Where A and B contain float4_e2m1 data, and scaleA/scaleB are UE8M0 scale factors.
    One scale factor per block of 32 K-elements (block_scale / .block32 mode).
    """

    def __init__(self, mma_m: int, mma_n: int, mma_k: int):
        super().__init__()
        self.mma_m = mma_m
        self.mma_n = mma_n
        self.mma_k = mma_k

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

        # Global views for fp4 data: A (M, K), B (N, K) stored row-major
        g_a = self.global_view(a_ptr, dtype=float4_e2m1, shape=[self.mma_m, self.mma_k])
        g_b = self.global_view(b_ptr, dtype=float4_e2m1, shape=[self.mma_n, self.mma_k])

        # Global views for scale factors: (128, 128) uint8 padded for tcgen05.copy compatibility
        # Actual scale data is in the first K/32 bytes of each row
        g_scale_a = self.global_view(scale_a_ptr, dtype=uint8, shape=[128, 128])
        g_scale_b = self.global_view(scale_b_ptr, dtype=uint8, shape=[128, 128])

        # Output: D (M, N) float16
        g_d = self.global_view(d_ptr, dtype=float16, shape=[self.mma_m, self.mma_n])

        # Shared memory for fp4 data
        s_a = self.shared_tensor(dtype=float4_e2m1, shape=[self.mma_m, self.mma_k])
        s_b = self.shared_tensor(dtype=float4_e2m1, shape=[self.mma_n, self.mma_k])

        # Shared memory for scale factors (padded to 128 bytes per row for tcgen05.copy)
        s_scale_a = self.shared_tensor(dtype=uint8, shape=[128, 128])
        s_scale_b = self.shared_tensor(dtype=uint8, shape=[128, 128])

        # Tensor memory: accumulator and scale factors
        t_d = self.tcgen05.alloc(dtype=float32, shape=[self.mma_m, self.mma_n])
        t_scale_a = self.tcgen05.alloc(dtype=uint8, shape=[128, 128])
        t_scale_b = self.tcgen05.alloc(dtype=uint8, shape=[128, 128])

        # Barriers: TMA data load, scale copy, MMA
        mbarriers = self.mbarrier.alloc(counts=[1, 1, 1])
        tma_mbarrier = mbarriers[0]
        scale_copy_mbarrier = mbarriers[1]
        mma_mbarrier = mbarriers[2]
        self.sync()

        # 1. Load fp4 data and scale factors from global to shared via TMA
        with self.single_thread():
            total_data_bytes = s_a.nbytes + s_b.nbytes + s_scale_a.nbytes + s_scale_b.nbytes
            self.mbarrier.arrive_and_expect_tx(tma_mbarrier, transaction_bytes=total_data_bytes)
            self.tma.global_to_shared(src=g_a, dst=s_a, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_b, dst=s_b, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_scale_a, dst=s_scale_a, offsets=[0, 0], mbarrier=tma_mbarrier)
            self.tma.global_to_shared(src=g_scale_b, dst=s_scale_b, offsets=[0, 0], mbarrier=tma_mbarrier)
        self.mbarrier.wait(tma_mbarrier, phase=0)

        # 2. Copy scale factors from shared to tensor memory
        with self.single_thread():
            self.tcgen05.copy(src=s_scale_a, dst=t_scale_a)
            self.tcgen05.copy(src=s_scale_b, dst=t_scale_b)
            self.tcgen05.commit(scale_copy_mbarrier)
        self.mbarrier.wait(scale_copy_mbarrier, phase=0)

        # 3. Perform block-scaled MMA: D = (A * scaleA) @ (B^T * scaleB)
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

        # 4. Load result from tensor memory and store to global
        r_d = self.tcgen05.load(t_d)
        self.tcgen05.wait_load()
        r_d_f16 = self.cast(r_d, dtype=float16)
        self.store_global(g_d, r_d_f16, offsets=[0, 0])

        # 5. Cleanup
        self.tcgen05.dealloc(t_d)
        self.tcgen05.dealloc(t_scale_a)
        self.tcgen05.dealloc(t_scale_b)


def _make_fp4_tensor(rows: int, cols: int, value_nibble: int = FP4_ONE) -> torch.Tensor:
    """Create a packed fp4 tensor where every element is the given nibble value."""
    packed_byte = (value_nibble << 4) | value_nibble
    num_bytes = rows * cols // 2
    return torch.full((num_bytes,), packed_byte, dtype=torch.uint8, device="cuda").reshape(rows, cols // 2)


def _make_scale_tensor(rows: int, scale_cols: int, value: int = UE8M0_ONE) -> torch.Tensor:
    """Create a padded (128, 128) uint8 scale tensor for tcgen05.copy compatibility.

    Actual scale data is in the first `scale_cols` bytes of each of the first `rows` rows.
    """
    t = torch.zeros(128, 128, dtype=torch.uint8, device="cuda")
    t[:rows, :scale_cols] = value
    return t


@tilus.testing.requires.nvgpu_sm100a
@pytest.mark.parametrize(
    "mma_m, mma_n, mma_k",
    [
        (128, 8, 64),
        (128, 16, 64),
        (128, 32, 64),
        (128, 8, 128),
        (128, 16, 128),
    ],
)
def test_tcgen05_scaled_mma(mma_m, mma_n, mma_k):
    """Test nvfp4 block-scaled MMA with all-ones fp4 data and unit scale factors.

    Expected: D[i,j] = K (since all A=1.0, B=1.0, scaleA=1.0, scaleB=1.0).
    """
    # Prepare fp4 data (all values = 1.0)
    a = _make_fp4_tensor(mma_m, mma_k)  # (M, K/2) uint8
    b = _make_fp4_tensor(mma_n, mma_k)  # (N, K/2) uint8

    # Prepare scale factors (all = 1.0 in UE8M0 format)
    scale_cols = mma_k // 32
    scale_a = _make_scale_tensor(mma_m, scale_cols)  # (128, 128) uint8
    scale_b = _make_scale_tensor(mma_n, scale_cols)  # (128, 128) uint8

    # Output
    d = torch.empty(mma_m, mma_n, dtype=torch.float16, device="cuda")

    # Run kernel
    kernel = Tcgen05ScaledMmaExample(mma_m, mma_n, mma_k)
    kernel(a, b, scale_a, scale_b, d)
    torch.cuda.synchronize()

    # Expected: each D[i,j] = sum over K of (1.0 * 1.0 * 1.0 * 1.0) = K
    expected = torch.full((mma_m, mma_n), float(mma_k), dtype=torch.float16, device="cuda")
    torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
