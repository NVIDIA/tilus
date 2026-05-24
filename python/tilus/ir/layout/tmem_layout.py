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
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from tilus.ir.node import IRNode


class TMemoryDuplication(Enum):
    """How a TMEM tensor's lane data is replicated across sub-partitions.

    SM100 TMEM has 128 lanes per CTA, organized as 4 sub-partitions of 32
    lanes. Some MMA kinds (e.g. block-scaled MMAs that read SFs) require
    operands to be replicated across sub-partitions; ``tcgen05.cp`` provides
    the multicast modes that produce these layouts. The enum value is the
    canonical name of the multicast pattern (matches the ``multicast=`` kwarg
    on :meth:`tilus.lang.cuda.Tcgen05Module.copy`).

    Attributes
    ----------
    NONE
        No replication; each lane holds unique data. Compatible with
        ``shape[0] in {32, 64, 128}`` (occupies that many physical lanes
        starting at ``lane_offset``).
    WARPX4
        32 unique lanes broadcast to all 4 sub-partitions; ``shape[0] == 32``.
    WARPX2_02_13
        64 unique lanes broadcast to two warp-pairs (warps {0,2} share half,
        warps {1,3} share the other half); ``shape[0] == 64``.
    WARPX2_01_23
        64 unique lanes broadcast to two warp-pairs (warps {0,1} share half,
        warps {2,3} share the other half); ``shape[0] == 64``.
    """

    NONE = "none"
    WARPX4 = "warpx4"
    WARPX2_02_13 = "warpx2_02_13"
    WARPX2_01_23 = "warpx2_01_23"


# Required ``shape[0]`` (unique lane count) for each duplication mode.
_DUPLICATION_LANE_SIZE: dict[TMemoryDuplication, set[int]] = {
    TMemoryDuplication.NONE: {32, 64, 128},
    TMemoryDuplication.WARPX4: {32},
    TMemoryDuplication.WARPX2_02_13: {64},
    TMemoryDuplication.WARPX2_01_23: {64},
}


@dataclass(frozen=True, eq=False)
class TMemoryLayout(IRNode):
    shape: tuple[int, ...]
    column_strides: tuple[int, ...]
    lane_offset: int
    duplication: TMemoryDuplication = TMemoryDuplication.NONE

    @staticmethod
    def create(
        shape: Sequence[int],
        column_strides: Sequence[int],
        lane_offset: int,
        duplication: TMemoryDuplication = TMemoryDuplication.NONE,
    ) -> TMemoryLayout:
        if len(shape) != len(column_strides):
            raise ValueError(
                "Dimension mismatch: shape has length {}, but column_strides has length {}".format(
                    len(shape), len(column_strides)
                )
            )
        if len(shape) < 2:
            raise ValueError("TMemLayout requires at least 2 dimensions, got {}".format(len(shape)))
        # Convention: shape[0] is the lane (row) dimension; all other dims are column-strided.
        if shape[0] not in [32, 64, 128]:
            raise ValueError("The number of rows (shape[0]) must be 32, 64, or 128, got {}".format(shape[0]))
        if column_strides[0] != 0:
            raise ValueError(
                "The column stride for the row dimension (column_strides[0]) must be 0, got {}".format(
                    column_strides[0]
                )
            )
        # Cross-validate shape[0] against the duplication mode.
        allowed_sizes = _DUPLICATION_LANE_SIZE[duplication]
        if shape[0] not in allowed_sizes:
            raise ValueError(
                "duplication={} requires shape[0] in {}, got shape[0]={}".format(
                    duplication.value, sorted(allowed_sizes), shape[0]
                )
            )
        # WARPX* modes always span all 128 physical lanes (replicated), so the
        # tensor must start at lane 0. lane_offset is only meaningful for NONE.
        if duplication != TMemoryDuplication.NONE and lane_offset != 0:
            raise ValueError("duplication={} requires lane_offset=0, got {}".format(duplication.value, lane_offset))
        # NONE: data fits in one CTA's TMEM lanes [lane_offset, lane_offset + shape[0]).
        if duplication == TMemoryDuplication.NONE and lane_offset + shape[0] > 128:
            raise ValueError(
                "lane_offset({}) + shape[0]({}) exceeds 128 physical TMEM lanes".format(lane_offset, shape[0])
            )
        return TMemoryLayout(
            shape=tuple(shape),
            column_strides=tuple(column_strides),
            lane_offset=lane_offset,
            duplication=duplication,
        )
