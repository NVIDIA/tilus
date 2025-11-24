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
from typing import Sequence

from tilus.ir.layout import TMemoryLayout


def tmemory_row_major(shape: Sequence[int]) -> TMemoryLayout:
    column_strides = []
    stride = 1
    for dim in reversed(range(len(shape))):
        if dim == len(shape) - 2:
            column_strides.append(0)
        else:
            column_strides.append(stride)
            stride *= shape[dim]
    column_strides = list(reversed(column_strides))
    return TMemoryLayout.create(shape, column_strides, lane_offset=0)


def tmemory_slice(
    tmem_layout: TMemoryLayout, lane_offset: int, slice_dims: Sequence[int], shape: Sequence[int]
) -> TMemoryLayout:
    lane_offset = tmem_layout.lane_offset + lane_offset
    strides = [tmem_layout.column_strides[dim] for dim in slice_dims]
    return TMemoryLayout.create(shape, strides, lane_offset)
