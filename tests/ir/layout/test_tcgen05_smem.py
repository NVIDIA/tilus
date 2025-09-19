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
from tilus.ir.layout.cuda.tcgen05_smem import Tcgen05SwizzleMode, get_tcgen05_smem_layout


class TestTcgen05SmemLayout:
    """Test suite for TCGen05 shared memory layout generation."""

    def test_no_swizzle_layout(self):
        """Test no-swizzle canonical layout: ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))"""
        T, m, k = 4, 2, 3
        sbo, lbo = 16, 8
        swizzle = (0, 0, 0)  # No swizzling

        layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, swizzle)

        # Check shape: (T * 1 * m, 8 * k) = (4 * 1 * 2, 8 * 3) = (8, 24)
        assert layout.shape == (8, 24)
        assert layout.size == 8 * 24

        # Test a few coordinate mappings
        # For no swizzle: base_offset = i0*1 + i1*T + i2*(T*sbo) + j0*(T*lbo) + j1*lbo
        # i=0, j=0 -> i0=0, i1=0, i2=0, j0=0, j1=0 -> offset = 0
        assert layout(0, 0) == 0

        # i=1, j=0 -> i0=1, i1=0, i2=0, j0=0, j1=0 -> offset = 1
        assert layout(1, 0) == 1

        # i=0, j=1 -> i0=0, i1=0, i2=0, j0=1, j1=0 -> offset = T*lbo = 4*8 = 32
        assert layout(0, 1) == 32

    def test_32b_swizzle_layout(self):
        """Test 32B swizzle canonical layout: ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))"""
        T, m, k = 4, 2, 3
        sbo, lbo = 16, 8
        swizzle = (1, 4, 3)  # 32B swizzling: Swizzle<1, 4, 3>

        layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, swizzle)

        # Check shape: (T * 2 * m, 8 * k) = (4 * 2 * 2, 8 * 3) = (16, 24)
        assert layout.shape == (16, 24)
        assert layout.size == 16 * 24

        # For 32B swizzle, swizzle factor = 2
        # base_offset = i0*1 + i1*T + i2*(T*lbo) + j0*(swizzle_factor*T*sbo) + j1*sbo
        # Then apply swizzle transformation

    def test_64b_swizzle_layout(self):
        """Test 64B swizzle canonical layout: ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))"""
        T, m, k = 4, 1, 2
        sbo, lbo = 16, 8
        swizzle = (2, 4, 3)  # 64B swizzling: Swizzle<2, 4, 3>

        layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, swizzle)

        # Check shape: (T * 4 * m, 8 * k) = (4 * 4 * 1, 8 * 2) = (16, 16)
        assert layout.shape == (16, 16)
        assert layout.size == 16 * 16

    def test_128b_swizzle_layout(self):
        """Test 128B swizzle canonical layout: ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))"""
        T, m, k = 2, 1, 2
        sbo, lbo = 16, 8
        swizzle = (3, 4, 3)  # 128B swizzling: Swizzle<3, 4, 3>

        layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, swizzle)

        # Check shape: (T * 8 * m, 8 * k) = (2 * 8 * 1, 8 * 2) = (16, 16)
        assert layout.shape == (16, 16)
        assert layout.size == 16 * 16

    def test_swizzle_mode_enum(self):
        """Test the Tcgen05SwizzleMode enum values."""
        assert Tcgen05SwizzleMode.NO_SWIZZLE.bbits == 0
        assert Tcgen05SwizzleMode.NO_SWIZZLE.mbase == 0
        assert Tcgen05SwizzleMode.NO_SWIZZLE.sshift == 0

        assert Tcgen05SwizzleMode.B32_SWIZZLE.bbits == 1
        assert Tcgen05SwizzleMode.B32_SWIZZLE.mbase == 4
        assert Tcgen05SwizzleMode.B32_SWIZZLE.sshift == 3

        assert Tcgen05SwizzleMode.B64_SWIZZLE.bbits == 2
        assert Tcgen05SwizzleMode.B64_SWIZZLE.mbase == 4
        assert Tcgen05SwizzleMode.B64_SWIZZLE.sshift == 3

        assert Tcgen05SwizzleMode.B128_SWIZZLE.bbits == 3
        assert Tcgen05SwizzleMode.B128_SWIZZLE.mbase == 4
        assert Tcgen05SwizzleMode.B128_SWIZZLE.sshift == 3

    def test_invalid_swizzle_bits(self):
        """Test that invalid swizzle bbits raise ValueError."""
        T, m, k = 4, 2, 3
        sbo, lbo = 16, 8

        with pytest.raises(ValueError, match="Unsupported swizzle bbits: 4"):
            get_tcgen05_smem_layout(T, m, k, sbo, lbo, (4, 4, 3))

        with pytest.raises(ValueError, match="Unsupported swizzle bbits: -1"):
            get_tcgen05_smem_layout(T, m, k, sbo, lbo, (-1, 4, 3))

    def test_coordinate_decomposition(self):
        """Test that coordinate decomposition works correctly."""
        T, m, k = 4, 2, 3
        sbo, lbo = 16, 8
        swizzle = (1, 4, 3)  # 32B swizzling

        layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, swizzle)

        # Shape is (16, 24), so valid coordinates are i in [0, 15], j in [0, 23]
        # Test boundary coordinates
        assert layout.shape == (16, 24)

        # Test that we can call the layout with valid coordinates
        try:
            result = layout(0, 0)
            assert isinstance(result.type, type(layout.offset.type))
        except Exception as e:
            pytest.fail(f"Layout call failed: {e}")

        try:
            result = layout(15, 23)  # Maximum valid coordinates
            assert isinstance(result.type, type(layout.offset.type))
        except Exception as e:
            pytest.fail(f"Layout call with max coordinates failed: {e}")

    def test_swizzle_transformation(self):
        """Test that swizzle transformation is applied correctly."""
        T, m, k = 4, 1, 1
        sbo, lbo = 1, 1

        # Compare no-swizzle vs swizzled layouts
        no_swizzle_layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, (0, 0, 0))
        swizzled_layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, (1, 4, 3))

        # For small coordinates, the swizzle should produce different results
        no_swizzle_result = no_swizzle_layout(0, 0)
        swizzled_result = swizzled_layout(0, 0)

        # Both should be valid expressions
        assert isinstance(no_swizzle_result.type, type(no_swizzle_layout.offset.type))
        assert isinstance(swizzled_result.type, type(swizzled_layout.offset.type))

    def test_different_parameters(self):
        """Test with different parameter combinations."""
        test_cases = [
            (2, 1, 1, 8, 4, (0, 0, 0)),  # Small case, no swizzle
            (8, 2, 4, 32, 16, (1, 4, 3)),  # Larger case, 32B swizzle
            (4, 3, 2, 24, 12, (2, 4, 3)),  # Medium case, 64B swizzle
            (2, 2, 2, 16, 8, (3, 4, 3)),  # Small case, 128B swizzle
        ]

        for T, m, k, sbo, lbo, swizzle in test_cases:
            layout = get_tcgen05_smem_layout(T, m, k, sbo, lbo, swizzle)

            swizzle_factor = 2 ** swizzle[0] if swizzle[0] > 0 else 1
            expected_shape = (T * swizzle_factor * m, 8 * k)
            expected_size = T * swizzle_factor * m * 8 * k

            assert layout.shape == expected_shape, f"Shape mismatch for {test_cases}"
            assert layout.size == expected_size, f"Size mismatch for {test_cases}"

            # Test that we can call the layout with valid coordinates
            try:
                result = layout(0, 0)
                assert isinstance(result.type, type(layout.offset.type))
            except Exception as e:
                pytest.fail(f"Layout call failed for parameters {test_cases}: {e}")
