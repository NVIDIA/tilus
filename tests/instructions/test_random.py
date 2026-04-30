# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for random number generation instructions (Philox-4x32 PRNG)."""

import numpy as np
import pytest
import tilus
import torch
from tilus import float32, int32, uint32, uint64
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.ops import spatial
from tilus.utils import cdiv


# Reference Philox-4x32-10 implementation for validation
def philox4x32_reference(seed: int, offset: np.ndarray, n_rounds: int = 10) -> tuple:
    """Reference Philox-4x32-10 in NumPy.

    Constants from Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (2011).
    """
    ROUND_A = np.uint32(0xD2511F53)
    ROUND_B = np.uint32(0xCD9E8D57)
    KEY_A = np.uint32(0x9E3779B9)
    KEY_B = np.uint32(0xBB67AE85)

    offset = offset.astype(np.uint32)
    c0 = offset.copy()
    c1 = np.zeros_like(offset, dtype=np.uint32)
    c2 = np.zeros_like(offset, dtype=np.uint32)
    c3 = np.zeros_like(offset, dtype=np.uint32)

    k0 = np.uint32(seed & 0xFFFFFFFF)
    k1 = np.uint32((seed >> 32) & 0xFFFFFFFF)

    def umulhi(a, b):
        """Unsigned multiply high for uint32."""
        return np.uint32((np.uint64(a) * np.uint64(b)) >> np.uint64(32))

    for _ in range(n_rounds):
        old_c0, old_c2 = c0.copy(), c2.copy()
        c0 = umulhi(ROUND_B, old_c2) ^ c1 ^ k0
        c2 = umulhi(ROUND_A, old_c0) ^ c3 ^ k1
        c1 = np.uint32(np.uint64(ROUND_B) * np.uint64(old_c2))
        c3 = np.uint32(np.uint64(ROUND_A) * np.uint64(old_c0))
        k0 = np.uint32(np.uint64(k0) + np.uint64(KEY_A))
        k1 = np.uint32(np.uint64(k1) + np.uint64(KEY_B))

    return c0, c1, c2, c3


class RandInt4xKernel(tilus.Script):
    """Kernel that generates 4 blocks of random uint32 using randint4x."""

    def __init__(self, block_size: int, layout: RegisterLayout, seed: int):
        super().__init__()
        self.block_size = block_size
        self.layout = layout
        self.num_warps = layout.spatial_size // 32
        self.seed = seed

    def __call__(
        self,
        n: int32,
        offset_ptr: ~uint32,
        out0_ptr: ~uint32,
        out1_ptr: ~uint32,
        out2_ptr: ~uint32,
        out3_ptr: ~uint32,
    ) -> None:
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (cdiv(n, self.block_size),)

        block_offset: int32 = self.blockIdx.x * self.block_size

        g_offset = self.global_view(offset_ptr, dtype=uint32, shape=(n,))
        g_out0 = self.global_view(out0_ptr, dtype=uint32, shape=(n,))
        g_out1 = self.global_view(out1_ptr, dtype=uint32, shape=(n,))
        g_out2 = self.global_view(out2_ptr, dtype=uint32, shape=(n,))
        g_out3 = self.global_view(out3_ptr, dtype=uint32, shape=(n,))

        r_offset = self.load_global(g_offset, offsets=[block_offset], shape=[self.block_size])
        self.annotate_layout(r_offset, self.layout)

        seed_expr = uint64(self.seed)
        r0, r1, r2, r3 = self.randint4x(seed=seed_expr, offset=r_offset)

        self.store_global(g_out0, r0, offsets=[block_offset])
        self.store_global(g_out1, r1, offsets=[block_offset])
        self.store_global(g_out2, r2, offsets=[block_offset])
        self.store_global(g_out3, r3, offsets=[block_offset])


class RandKernel(tilus.Script):
    """Kernel that generates random float32 in [0, 1) using rand."""

    def __init__(self, block_size: int, layout: RegisterLayout, seed: int):
        super().__init__()
        self.block_size = block_size
        self.layout = layout
        self.num_warps = layout.spatial_size // 32
        self.seed = seed

    def __call__(
        self,
        n: int32,
        offset_ptr: ~uint32,
        out_ptr: ~float32,
    ) -> None:
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (cdiv(n, self.block_size),)

        block_offset: int32 = self.blockIdx.x * self.block_size

        g_offset = self.global_view(offset_ptr, dtype=uint32, shape=(n,))
        g_out = self.global_view(out_ptr, dtype=float32, shape=(n,))

        r_offset = self.load_global(g_offset, offsets=[block_offset], shape=[self.block_size])
        self.annotate_layout(r_offset, self.layout)

        seed_expr = uint64(self.seed)
        r_out = self.rand(seed=seed_expr, offset=r_offset)

        self.store_global(g_out, r_out, offsets=[block_offset])


class RandnKernel(tilus.Script):
    """Kernel that generates random float32 ~ N(0, 1) using randn."""

    def __init__(self, block_size: int, layout: RegisterLayout, seed: int):
        super().__init__()
        self.block_size = block_size
        self.layout = layout
        self.num_warps = layout.spatial_size // 32
        self.seed = seed

    def __call__(
        self,
        n: int32,
        offset_ptr: ~uint32,
        out_ptr: ~float32,
    ) -> None:
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (cdiv(n, self.block_size),)

        block_offset: int32 = self.blockIdx.x * self.block_size

        g_offset = self.global_view(offset_ptr, dtype=uint32, shape=(n,))
        g_out = self.global_view(out_ptr, dtype=float32, shape=(n,))

        r_offset = self.load_global(g_offset, offsets=[block_offset], shape=[self.block_size])
        self.annotate_layout(r_offset, self.layout)

        seed_expr = uint64(self.seed)
        r_out = self.randn(seed=seed_expr, offset=r_offset)

        self.store_global(g_out, r_out, offsets=[block_offset])


BLOCK_SIZE = 128
LAYOUT = spatial(128)
SEED = 42


def _make_uint32_tensor(n: int, device="cuda") -> torch.Tensor:
    """Create a uint32 tensor by viewing an int32 tensor."""
    return torch.zeros(n, dtype=torch.int32, device=device).view(torch.uint32)


def _arange_uint32(n: int, device="cuda") -> torch.Tensor:
    """Create arange as uint32 by viewing int32."""
    return torch.arange(n, dtype=torch.int32, device=device).view(torch.uint32)


@pytest.mark.parametrize("n", [128, 1024])
def test_randint4x_correctness(n: int) -> None:
    """Test that randint4x matches the reference Philox implementation."""
    offsets = _arange_uint32(n)

    out0 = _make_uint32_tensor(n)
    out1 = _make_uint32_tensor(n)
    out2 = _make_uint32_tensor(n)
    out3 = _make_uint32_tensor(n)

    kernel = RandInt4xKernel(BLOCK_SIZE, LAYOUT, SEED)
    kernel(n, offsets, out0, out1, out2, out3)

    # Reference
    offsets_np = np.arange(n, dtype=np.uint32)
    ref0, ref1, ref2, ref3 = philox4x32_reference(SEED, offsets_np)

    np.testing.assert_array_equal(out0.cpu().numpy().view(np.uint32), ref0)
    np.testing.assert_array_equal(out1.cpu().numpy().view(np.uint32), ref1)
    np.testing.assert_array_equal(out2.cpu().numpy().view(np.uint32), ref2)
    np.testing.assert_array_equal(out3.cpu().numpy().view(np.uint32), ref3)


@pytest.mark.parametrize("n", [4096])
def test_rand_distribution(n: int) -> None:
    """Test that rand produces values in [0, 1) with roughly uniform distribution."""
    offsets = _arange_uint32(n)
    out = torch.empty(n, dtype=torch.float32, device="cuda")

    kernel = RandKernel(BLOCK_SIZE, LAYOUT, SEED)
    kernel(n, offsets, out)

    out_cpu = out.cpu().numpy()

    # All values should be in [0, 1)
    assert np.all(out_cpu >= 0.0), "rand produced values < 0"
    assert np.all(out_cpu < 1.0), "rand produced values >= 1"

    # Rough uniformity check: mean should be close to 0.5
    mean = np.mean(out_cpu)
    assert 0.3 < mean < 0.7, f"rand mean {mean} is too far from 0.5"


@pytest.mark.parametrize("n", [4096])
def test_randn_distribution(n: int) -> None:
    """Test that randn produces values with roughly normal distribution."""
    offsets = _arange_uint32(n)
    out = torch.empty(n, dtype=torch.float32, device="cuda")

    kernel = RandnKernel(BLOCK_SIZE, LAYOUT, SEED)
    kernel(n, offsets, out)

    out_cpu = out.cpu().numpy()

    # Mean should be close to 0
    mean = np.mean(out_cpu)
    assert -0.5 < mean < 0.5, f"randn mean {mean} is too far from 0"

    # Std should be close to 1
    std = np.std(out_cpu)
    assert 0.5 < std < 1.5, f"randn std {std} is too far from 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
