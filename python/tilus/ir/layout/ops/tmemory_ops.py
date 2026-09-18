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
from typing import Sequence

from tilus.ir.layout import TMemoryLayout
from tilus.ir.layout.tmem_layout import TMemoryDuplication


def tmemory_row_major(shape: Sequence[int], duplication: TMemoryDuplication = TMemoryDuplication.NONE) -> TMemoryLayout:
    # Convention: dim 0 is the lane axis (stride 0); dims 1..-1 are column-strided
    # in row-major order (innermost dim has stride 1).
    column_strides = [0] * len(shape)
    stride = 1
    for dim in reversed(range(1, len(shape))):
        column_strides[dim] = stride
        stride *= shape[dim]
    return TMemoryLayout.create(shape, column_strides, lane_offset=0, duplication=duplication)


def tmemory_slice(
    tmem_layout: TMemoryLayout, lane_offset: int, slice_dims: Sequence[int], shape: Sequence[int]
) -> TMemoryLayout:
    # Column-dim slicing preserves the parent's duplication and lane_offset.
    # Lane-dim slicing (shape[0] differs from parent's shape[0]) is only legal
    # for NONE-duplicated tensors and adds the requested lane_offset shift.
    new_lane_offset = tmem_layout.lane_offset + lane_offset
    if shape[0] != tmem_layout.shape[0] and tmem_layout.duplication != TMemoryDuplication.NONE:
        raise ValueError(
            "Lane-dim slicing of a TMEM tensor with duplication={} is not supported "
            "(parent shape[0]={}, sliced shape[0]={}); only NONE-duplicated tensors "
            "may have their lane dim sliced.".format(tmem_layout.duplication.value, tmem_layout.shape[0], shape[0])
        )
    if tmem_layout.duplication != TMemoryDuplication.NONE and lane_offset != 0:
        raise ValueError(
            "duplication={} disallows non-zero lane_offset shift, got {}".format(
                tmem_layout.duplication.value, lane_offset
            )
        )
    strides = [tmem_layout.column_strides[dim] for dim in slice_dims]
    return TMemoryLayout.create(shape, strides, new_lane_offset, duplication=tmem_layout.duplication)
