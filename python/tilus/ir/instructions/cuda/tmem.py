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
import functools

from dataclasses import dataclass
from typing import Sequence

from hidet.ir.dtypes import uint32
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.layout.register_layout import RegisterLayout, visualize_layout
from tilus.ir.layout.register_layout_ops import spatial, register_layout, local
from tilus.ir.tensor import RegisterTensor, TMemoryTensor
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import Tcgen05LoadStoreShapeKind, Tcgen05LoadStoreNumKind


@functools.cache
def get_ldst_layout(shape: Tcgen05LoadStoreShapeKind) -> RegisterLayout:

    # see https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-layout
    if shape == Tcgen05LoadStoreShapeKind.R32x32B:
        return spatial(32, 1)
    elif shape == Tcgen05LoadStoreShapeKind.R16x64B:
        """
        ┌───────┬───────┐
        │ 0: 0  │ 2: 0  │
        ├───────┼───────┤
        │ 4: 0  │ 6: 0  │
        ├───────┼───────┤
        │ 8: 0  │ 10: 0 │
        ├───────┼───────┤
        │ 12: 0 │ 14: 0 │
        ├───────┼───────┤
        │ 16: 0 │ 18: 0 │
        ├───────┼───────┤
        │ 20: 0 │ 22: 0 │
        ├───────┼───────┤
        │ 24: 0 │ 26: 0 │
        ├───────┼───────┤
        │ 28: 0 │ 30: 0 │
        ├───────┼───────┤
        │ 1: 0  │ 3: 0  │
        ├───────┼───────┤
        │ 5: 0  │ 7: 0  │
        ├───────┼───────┤
        │ 9: 0  │ 11: 0 │
        ├───────┼───────┤
        │ 13: 0 │ 15: 0 │
        ├───────┼───────┤
        │ 17: 0 │ 19: 0 │
        ├───────┼───────┤
        │ 21: 0 │ 23: 0 │
        ├───────┼───────┤
        │ 25: 0 │ 27: 0 │
        ├───────┼───────┤
        │ 29: 0 │ 31: 0 │
        └───────┴───────┘
        """
        return register_layout(shape=[16, 2], mode_shape=[2, 8, 2], spatial_modes=[1, 2, 0], local_modes=[])
    elif shape == Tcgen05LoadStoreShapeKind.R16x128B:
        return local(2, 1).spatial(8, 4)
    elif shape == Tcgen05LoadStoreShapeKind.R16x256B:
        return local(2, 1).spatial(8, 4).local(1, 2)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    


@dataclass(frozen=True, eq=False)
class TMemoryAllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(num_columns: int, cta_group: int) -> TMemoryAllocInst:
        assert cta_group in (1, 2)
        output = TMemoryTensor.create(dtype=uint32, shape=[128, num_columns], first_lane=0)
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


if __name__ == "__main__":
    layout_0 = spatial(32, 1)
    layout_1 = register_layout(shape=[16, 2], mode_shape=[2, 8, 2], spatial_modes=[1, 2, 0], local_modes=[])
    layout_3 = get_ldst_layout(Tcgen05LoadStoreShapeKind.R16x64B, Tcgen05LoadStoreNumKind.X2)
    print(layout_3)
    print(visualize_layout(layout_0))
    print(visualize_layout(layout_1))
    print(visualize_layout(layout_3))



