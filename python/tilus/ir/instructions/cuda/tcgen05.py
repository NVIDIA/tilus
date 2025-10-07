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

from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor


@dataclass(frozen=True, eq=False)
class Tcgen05AllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(dtype: DataType, shape: Sequence[int], cta_group: int) -> Tcgen05AllocInst:
        assert cta_group in (1, 2)
        assert len(shape) == 2
        assert shape[0] == 128
        assert shape[1] * dtype.nbits % 32 == 0
        num_columns = shape[1] * dtype.nbits // 32
        assert num_columns % 32 == 0 and 32 <= num_columns <= 512
        output = TMemoryTensor.create(dtype=dtype, shape=shape, first_lane=0)
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
class Tcgen05SliceInst(Instruction):
    offsets: tuple[int, int]

    @staticmethod
    def create(tmem: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> Tcgen05SliceInst:
        assert len(offsets) == len(shape) == 2
        for o, s, ts in zip(offsets, shape, tmem.shape):
            assert 0 <= o < ts
            assert 0 < s <= ts - o
        output = TMemoryTensor.create(dtype=tmem.dtype, shape=shape, first_lane=tmem.first_lane + offsets[0])
        return Tcgen05SliceInst(output=output, inputs=(tmem,), offsets=(offsets[0], offsets[1]))


@dataclass(frozen=True, eq=False)
class Tcgen05ViewInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> Tcgen05ViewInst:
        if len(shape) != 2:
            raise ValueError("Only 2D shape is supported.")
        if shape[0] != tmem.shape[0]:
            raise ValueError("The first dimension must be the same as the original tensor.")
        if dtype.nbits * shape[1] != tmem.dtype.nbits * tmem.shape[1]:
            raise ValueError("The total number of bits must be the same as the original tensor.")

        output = TMemoryTensor.create(dtype=dtype, shape=shape, first_lane=tmem.first_lane)
        return Tcgen05ViewInst(output=output, inputs=(tmem,))


@dataclass(frozen=True, eq=False)
class Tcgen05LoadInst(Instruction):
    offsets: tuple[int, int]

    @staticmethod
    def create(tmem: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> Tcgen05LoadInst:
        assert len(offsets) == len(shape) == 2
        for o, s, ts in zip(offsets, shape, tmem.shape):
            assert 0 <= o < ts
            assert 0 < s <= ts - o
        output = RegisterTensor.create(dtype=tmem.dtype, shape=shape)
        return Tcgen05LoadInst(output=output, inputs=(tmem,), offsets=(offsets[0], offsets[1]))


@dataclass(frozen=True, eq=False)
class Tcgen05StoreInst(Instruction):
    offsets: tuple[int, int]

    @staticmethod
    def create(tmem: TMemoryTensor, src: RegisterTensor, offsets: Sequence[int]) -> Tcgen05StoreInst:
        assert len(offsets) == 2
        for o, s, ts in zip(offsets, src.shape, tmem.shape):
            assert 0 <= o < ts
            assert 0 < s <= ts - o

        return Tcgen05StoreInst(output=None, inputs=(tmem, src), offsets=(offsets[0], offsets[1]))


@dataclass(frozen=True, eq=False)
class Tcgen05WaitInst(Instruction):
    wait_load: bool
    wait_store: bool

    @staticmethod
    def create(wait_load: bool, wait_store: bool) -> Tcgen05WaitInst:
        return Tcgen05WaitInst(output=None, inputs=(), wait_load=wait_load, wait_store=wait_store)


@dataclass(frozen=True, eq=False)
class Tcgen05CopyInst(Instruction):
    @staticmethod
    def create(src: SharedTensor, dst: TMemoryTensor) -> Tcgen05CopyInst:
        return Tcgen05CopyInst(output=None, inputs=(dst, src))


@dataclass(frozen=True, eq=False)
class Tcgen05CommitInst(Instruction):
    mbarrier: Expr
    cta_mask: Optional[int]

    @staticmethod
    def create(mbarrier: Expr, cta_mask: Optional[int] = None) -> Tcgen05CommitInst:
        return Tcgen05CommitInst(output=None, inputs=(), mbarrier=mbarrier, cta_mask=cta_mask)


@dataclass(frozen=True, eq=False)
class Tcgen05MmaSSInst(Instruction):

    @staticmethod
    def create(
        a: SharedTensor,
        b: SharedTensor,
        d: TMemoryTensor,
    ) -> Tcgen05MmaSSInst:
        return Tcgen05MmaSSInst(output=None, inputs=(a, b, d))


@dataclass(frozen=True, eq=False)
class Tcgen05MmaTSInst(Instruction):

    @staticmethod
    def create(
        a: TMemoryTensor,
        b: SharedTensor,
        d: TMemoryTensor,
    ) -> Tcgen05MmaTSInst:
        return Tcgen05MmaTSInst(output=None, inputs=(a, b, d))
