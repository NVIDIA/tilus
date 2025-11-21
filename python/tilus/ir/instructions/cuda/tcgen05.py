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

from hidet.ir.expr import Constant, Expr
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor


@dataclass(frozen=True, eq=False)
class Tcgen05AllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(dtype: DataType, shape: Sequence[int], cta_group: int) -> Tcgen05AllocInst:
        assert len(shape) >= 2, "Tcgen05AllocInst only supports tensors with rank >= 2."
        assert shape[-2] in (32, 64, 128), "The second last dimension must be 32, 64, or 128."
        output = TMemoryTensor.create(dtype=dtype, shape=shape)
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
    offsets: tuple[Expr, ...]
    slice_dims: tuple[int, ...]

    @staticmethod
    def create(
        tmem: TMemoryTensor,
        offsets: Sequence[Expr],
        slice_dims: Sequence[int],
        slice_shape: Sequence[int],
    ) -> Tcgen05SliceInst:
        assert len(tmem.shape) == len(offsets)
        assert len(slice_shape) == len(slice_dims)
        assert len(slice_dims) >= 2 and all(len(tmem.shape) - 1 - i in slice_dims for i in range(2)), (
            "The last two dimensions must be included in the slice."
        )
        assert isinstance(offsets[-2], Constant), "The row-offset must be a constant."
        output = TMemoryTensor.create(dtype=tmem.dtype, shape=slice_shape)
        return Tcgen05SliceInst(output=output, inputs=(tmem,), offsets=tuple(offsets), slice_dims=tuple(slice_dims))


@dataclass(frozen=True, eq=False)
class Tcgen05ViewInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> Tcgen05ViewInst:
        if len(tmem.shape) != len(shape):
            raise ValueError("The rank of the new shape must match the original shape.")
        if any(s1 != s2 for s1, s2 in zip(tmem.shape[:-1], shape[:-1])):
            raise ValueError("All dimensions except the last one must match in the view operation.")
        if tmem.shape[-1] * tmem.dtype.nbits != shape[-1] * dtype.nbits:
            raise ValueError(
                "The total number of bits in the last dimension must remain the same in the view operation."
            )
        output = TMemoryTensor.create(dtype=dtype, shape=shape)
        return Tcgen05ViewInst(output=output, inputs=(tmem,))


@dataclass(frozen=True, eq=False)
class Tcgen05LoadInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor) -> Tcgen05LoadInst:
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.load)
        output = RegisterTensor.create(dtype=tmem.dtype, shape=tmem.shape)
        return Tcgen05LoadInst(output=output, inputs=(tmem,))


@dataclass(frozen=True, eq=False)
class Tcgen05StoreInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor, src: RegisterTensor) -> Tcgen05StoreInst:
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.store)
        return Tcgen05StoreInst(output=None, inputs=(tmem, src))


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
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.copy)
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
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.mma)
        return Tcgen05MmaSSInst(output=None, inputs=(a, b, d))


@dataclass(frozen=True, eq=False)
class Tcgen05MmaTSInst(Instruction):
    @staticmethod
    def create(
        a: TMemoryTensor,
        b: SharedTensor,
        d: TMemoryTensor,
    ) -> Tcgen05MmaTSInst:
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.mma)
        return Tcgen05MmaTSInst(output=None, inputs=(a, b, d))
