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
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tilus.ir.node import IRNode


@dataclass(frozen=True, eq=False)
class TMemoryLayout(IRNode):
    shape: tuple[int, ...]
    column_strides: tuple[int, ...]
    lane_offset: int

    @staticmethod
    def create(shape: Sequence[int], column_strides: Sequence[int], lane_offset: int) -> TMemoryLayout:
        if len(shape) != len(column_strides):
            raise ValueError(
                "Dimension mismatch: shape has length {}, but column_strides has length {}".format(
                    len(shape), len(column_strides)
                )
            )
        if len(shape) < 2:
            raise ValueError("TMemLayout requires at least 2 dimensions, got {}".format(len(shape)))
        if shape[-2] not in [32, 64, 128]:
            raise ValueError("The number of rows (shape[-2]) must be 32, 64, or 128, got {}".format(shape[-2]))
        if column_strides[-2] != 0:
            raise ValueError(
                "The column stride for the row dimension (column_strides[-2]) must be 0, got {}".format(
                    column_strides[-2]
                )
            )
        return TMemoryLayout(shape=tuple(shape), column_strides=tuple(column_strides), lane_offset=lane_offset)
