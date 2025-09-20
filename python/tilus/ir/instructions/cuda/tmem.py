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
from typing import Optional, Sequence

from hidet.ir.expr import Expr
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor


@dataclass(frozen=True, eq=False)
class TMemoryAllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(dtype: DataType, shape: Sequence[int], cta_group: int) -> TMemoryAllocInst:
        assert cta_group in (1, 2)
        assert len(shape) == 2
        assert shape[0] == 128
        assert shape[1] * dtype.nbits % 32 == 0
        num_columns = shape[1] * dtype.nbits // 32
        assert num_columns % 32 == 0 and 32 <= num_columns <= 512
        output = TMemoryTensor.create(dtype=dtype, shape=shape, first_lane=0)
        return TMemoryAllocInst(output=output, inputs=(), cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class TMemoryDeallocInst(Instruction):
    @staticmethod
    def create(tmt: TMemoryTensor) -> TMemoryDeallocInst:
        return TMemoryDeallocInst(output=None, inputs=(tmt,))


@dataclass(frozen=True, eq=False)
class TMemoryRelinquishAllocPermitInst(Instruction):
    cta_group: int = 1

    @staticmethod
    def create(cta_group: int) -> TMemoryRelinquishAllocPermitInst:
        return TMemoryRelinquishAllocPermitInst(output=None, inputs=(), cta_group=cta_group)


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


@dataclass(frozen=True, eq=False)
class TMemoryLoadInst(Instruction):
    offsets: tuple[int, int]

    @staticmethod
    def create(tmem: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> TMemoryLoadInst:
        assert len(offsets) == len(shape) == 2
        for o, s, ts in zip(offsets, shape, tmem.shape):
            assert 0 <= o < ts
            assert 0 < s <= ts - o
        output = RegisterTensor.create(dtype=tmem.dtype, shape=shape)
        return TMemoryLoadInst(output=output, inputs=(tmem,), offsets=(offsets[0], offsets[1]))


@dataclass(frozen=True, eq=False)
class TMemoryStoreInst(Instruction):
    offsets: tuple[int, int]

    @staticmethod
    def create(tmem: TMemoryTensor, src: RegisterTensor, offsets: Sequence[int]) -> TMemoryStoreInst:
        assert len(offsets) == 2
        for o, s, ts in zip(offsets, src.shape, tmem.shape):
            assert 0 <= o < ts
            assert 0 < s <= ts - o

        return TMemoryStoreInst(output=None, inputs=(tmem, src), offsets=(offsets[0], offsets[1]))


@dataclass(frozen=True, eq=False)
class TMemoryWaitInst(Instruction):
    wait_load: bool
    wait_store: bool

    @staticmethod
    def create(wait_load: bool, wait_store: bool) -> TMemoryWaitInst:
        return TMemoryWaitInst(output=None, inputs=(), wait_load=wait_load, wait_store=wait_store)


@dataclass(frozen=True, eq=False)
class Tcgen05CopyInst(Instruction):
    @staticmethod
    def create(src: SharedTensor, dst: TMemoryTensor) -> Tcgen05CopyInst:
        return Tcgen05CopyInst(output=None, inputs=(dst, src))


@dataclass(frozen=True, eq=False)
class TMemoryCommitInst(Instruction):
    mbarrier: Expr
    cta_mask: Optional[int]

    @staticmethod
    def create(mbarrier: Expr, cta_mask: Optional[int] = None) -> TMemoryCommitInst:
        return TMemoryCommitInst(output=None, inputs=(), mbarrier=mbarrier, cta_mask=cta_mask)
