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
"""Unit tests for :class:`TMemoryLayout` shape / duplication / lane_offset cross-validation."""

import pytest
from tilus.ir.layout import TMemoryLayout
from tilus.ir.layout.ops.tmemory_ops import tmemory_row_major, tmemory_slice
from tilus.ir.layout.tmem_layout import TMemoryDuplication

# ---------------------------------------------------------------------------
# Construction: shape, column_strides, lane_offset, duplication
# ---------------------------------------------------------------------------


class TestCreate:
    def test_default_duplication_is_none(self):
        layout = TMemoryLayout.create(shape=(128, 64), column_strides=(0, 1), lane_offset=0)
        assert layout.duplication == TMemoryDuplication.NONE

    @pytest.mark.parametrize("lane_size", [32, 64, 128])
    def test_none_duplication_accepts_all_lane_sizes(self, lane_size):
        layout = TMemoryLayout.create(
            shape=(lane_size, 16),
            column_strides=(0, 1),
            lane_offset=0,
            duplication=TMemoryDuplication.NONE,
        )
        assert layout.shape[0] == lane_size

    def test_warpx4_requires_lane_32(self):
        # WARPX4 with shape[0]=32 — ok
        TMemoryLayout.create(
            shape=(32, 4),
            column_strides=(0, 1),
            lane_offset=0,
            duplication=TMemoryDuplication.WARPX4,
        )
        # WARPX4 with shape[0]=64 — illegal
        with pytest.raises(ValueError, match="WARPX4|warpx4"):
            TMemoryLayout.create(
                shape=(64, 4),
                column_strides=(0, 1),
                lane_offset=0,
                duplication=TMemoryDuplication.WARPX4,
            )
        # WARPX4 with shape[0]=128 — illegal
        with pytest.raises(ValueError, match="WARPX4|warpx4"):
            TMemoryLayout.create(
                shape=(128, 4),
                column_strides=(0, 1),
                lane_offset=0,
                duplication=TMemoryDuplication.WARPX4,
            )

    @pytest.mark.parametrize("duplication", [TMemoryDuplication.WARPX2_02_13, TMemoryDuplication.WARPX2_01_23])
    def test_warpx2_requires_lane_64(self, duplication):
        # legal: shape[0] == 64
        TMemoryLayout.create(shape=(64, 4), column_strides=(0, 1), lane_offset=0, duplication=duplication)
        # illegal: shape[0] == 32 or 128
        for bad in (32, 128):
            with pytest.raises(ValueError):
                TMemoryLayout.create(
                    shape=(bad, 4),
                    column_strides=(0, 1),
                    lane_offset=0,
                    duplication=duplication,
                )

    @pytest.mark.parametrize(
        "duplication",
        [TMemoryDuplication.WARPX4, TMemoryDuplication.WARPX2_02_13, TMemoryDuplication.WARPX2_01_23],
    )
    def test_duplicated_layout_disallows_nonzero_lane_offset(self, duplication):
        lane_size = 32 if duplication == TMemoryDuplication.WARPX4 else 64
        with pytest.raises(ValueError, match="lane_offset"):
            TMemoryLayout.create(
                shape=(lane_size, 4),
                column_strides=(0, 1),
                lane_offset=32,
                duplication=duplication,
            )

    def test_none_duplication_lane_offset_must_fit_in_128(self):
        # legal: lane_offset + shape[0] <= 128
        TMemoryLayout.create(shape=(64, 4), column_strides=(0, 1), lane_offset=64)
        TMemoryLayout.create(shape=(32, 4), column_strides=(0, 1), lane_offset=96)
        # illegal: overflows 128
        with pytest.raises(ValueError, match="exceeds 128"):
            TMemoryLayout.create(shape=(64, 4), column_strides=(0, 1), lane_offset=96)
        with pytest.raises(ValueError, match="exceeds 128"):
            TMemoryLayout.create(shape=(128, 4), column_strides=(0, 1), lane_offset=1)

    def test_invalid_lane_size(self):
        with pytest.raises(ValueError, match="must be 32, 64, or 128"):
            TMemoryLayout.create(shape=(16, 4), column_strides=(0, 1), lane_offset=0)

    def test_dim_mismatch(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            TMemoryLayout.create(shape=(128, 4), column_strides=(0,), lane_offset=0)

    def test_lane_must_have_zero_stride(self):
        with pytest.raises(ValueError, match="column_strides\\[0\\]"):
            TMemoryLayout.create(shape=(128, 4), column_strides=(1, 1), lane_offset=0)

    def test_higher_rank(self):
        # rank 3: lane=128, then two column-strided dims
        layout = TMemoryLayout.create(shape=(128, 4, 8), column_strides=(0, 8, 1), lane_offset=0)
        assert layout.shape == (128, 4, 8)
        assert layout.column_strides == (0, 8, 1)


# ---------------------------------------------------------------------------
# tmemory_row_major
# ---------------------------------------------------------------------------


class TestRowMajor:
    def test_default_none_duplication(self):
        layout = tmemory_row_major(shape=(128, 64))
        assert layout.duplication == TMemoryDuplication.NONE
        assert layout.column_strides == (0, 1)

    def test_warpx4_via_row_major(self):
        layout = tmemory_row_major(shape=(32, 4, 4), duplication=TMemoryDuplication.WARPX4)
        assert layout.duplication == TMemoryDuplication.WARPX4
        # row-major columns: dim 2 stride=1, dim 1 stride=4, dim 0 stride=0 (lane)
        assert layout.column_strides == (0, 4, 1)

    def test_higher_rank_strides(self):
        layout = tmemory_row_major(shape=(128, 2, 3, 4))
        # rightmost stride = 1, then 4, then 12; lane stride = 0
        assert layout.column_strides == (0, 12, 4, 1)


# ---------------------------------------------------------------------------
# tmemory_slice
# ---------------------------------------------------------------------------


class TestSlice:
    def test_column_slice_preserves_duplication(self):
        # parent: WARPX4 [32, M_fold=4, K=4]
        parent = tmemory_row_major(shape=(32, 4, 4), duplication=TMemoryDuplication.WARPX4)
        # slice at M_fold=1, keep [32, 4]
        child = tmemory_slice(parent, lane_offset=0, slice_dims=[0, 2], shape=(32, 4))
        assert child.duplication == TMemoryDuplication.WARPX4
        assert child.shape == (32, 4)
        assert child.lane_offset == 0

    def test_lane_slice_only_for_none(self):
        # NONE [128, 64]  ->  [64, 64] starting at lane 64 — legal
        parent = tmemory_row_major(shape=(128, 64))
        child = tmemory_slice(parent, lane_offset=64, slice_dims=[0, 1], shape=(64, 64))
        assert child.duplication == TMemoryDuplication.NONE
        assert child.shape == (64, 64)
        assert child.lane_offset == 64

    def test_lane_slice_warpx4_raises(self):
        parent = tmemory_row_major(shape=(32, 4, 4), duplication=TMemoryDuplication.WARPX4)
        # try to lane-slice (would change shape[0] from 32 to a different value
        # OR keep 32 but apply non-zero lane_offset — both illegal for WARPX4).
        with pytest.raises(ValueError, match="duplication=warpx4|lane_offset"):
            tmemory_slice(parent, lane_offset=32, slice_dims=[0, 1, 2], shape=(32, 4, 4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
