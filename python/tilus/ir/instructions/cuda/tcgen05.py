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

from hidet.ir.dtypes import uint32
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.tensor import TMemoryTensor


@dataclass(frozen=True, eq=False)
class Tcgen05AllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(num_columns: int, cta_group: int) -> Tcgen05AllocInst:
        assert cta_group in (1, 2)
        output = TMemoryTensor.create(dtype=uint32, shape=[128, num_columns], first_lane=0)
        return Tcgen05AllocInst(output=output, inputs=(), cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class Tcgen05DeallocInst(Instruction):
    @staticmethod
    def create(tmt: TMemoryTensor) -> Tcgen05DeallocInst:
        return Tcgen05DeallocInst(output=None, inputs=(tmt,))


@dataclass(frozen=True, eq=False)
class Tcgen05RelinquishAllocPermitInst(Instruction):
    cta_group: int = 1

    @staticmethod
    def create(cta_group: int) -> Tcgen05RelinquishAllocPermitInst:
        return Tcgen05RelinquishAllocPermitInst(output=None, inputs=(), cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class TMemorySliceInst(Instruction):
    offsets: tuple[int, int]

    @staticmethod
    def create(tmem: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> TMemorySliceInst:
        assert len(offsets) == len(shape) == 2
        for o, s, ts in zip(offsets, shape, tmem.shape):
            assert 0 <= o < ts
            assert 0 < s <= ts - o
        output = TMemoryTensor.create(dtype=tmem.dtype, shape=shape, first_lane=tmem.first_lane + offsets[0])
        return TMemorySliceInst(output=output, inputs=(tmem,), offsets=(offsets[0], offsets[1]))


@dataclass(frozen=True, eq=False)
class TMemoryViewInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> TMemoryViewInst:
        if len(shape) != 2:
            raise ValueError("Only 2D shape is supported.")
        if shape[0] != tmem.shape[0]:
            raise ValueError("The first dimension must be the same as the original tensor.")
        if dtype.nbits * shape[1] != tmem.dtype.nbits * tmem.shape[1]:
            raise ValueError("The total number of bits must be the same as the original tensor.")

        output = TMemoryTensor.create(dtype=dtype, shape=shape, first_lane=tmem.first_lane)
        return TMemoryViewInst(output=output, inputs=(tmem,))
