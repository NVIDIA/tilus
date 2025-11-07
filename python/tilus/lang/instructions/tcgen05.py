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
from typing import Optional, Sequence
from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from tilus.ir.tensor import TMemoryTensor, RegisterTensor, SharedTensor
from tilus.ir.builders import StmtBuilder
from tilus.ir.inst import InstructionError
from tilus.utils import is_power_of_two

from .root import InstructionGroup


class Tcgen05InstructionGroup(InstructionGroup):
    def alloc(
        self, dtype: DataType, shape: Sequence[int], cta_group: int = 1, init: Optional[Expr | float | int] = None
    ) -> TMemoryTensor:
        if cta_group not in [1, 2]:
            raise InstructionError("cta_group must be 1 or 2")
        if len(shape) != 2:
            raise InstructionError("shape must be a sequence of length 2, got {}".format(shape))
        if shape[0] != 128:
            raise InstructionError("shape[0] must be 128, got {}".format(shape[0]))
        if dtype.nbits > 32 or 32 % dtype.nbits != 0:
            raise InstructionError("dtype must be 8-bit, 16-bit, or 32-bit, got {}".format(dtype))
        num_columns = shape[1] * dtype.nbits // 32
        if not is_power_of_two(num_columns) or num_columns < 32 or num_columns > 512:
            raise InstructionError(
                "num_columns must be a power of two and in the range [32, 512], got {}".format(num_columns)
            )
        ret = self._builder.tcgen05_alloc(dtype, shape, cta_group)
        if init is not None:
            self._builder.tcgen05_store(
                ret,
                src=self._builder.allocate_register(dtype=dtype, shape=shape, f_init=lambda _: dtype(init)),
                offsets=[0, 0],
            )
            self._builder.tcgen05_wait_store()
        return ret

    def dealloc(self, tensor: TMemoryTensor) -> None:
        self._builder.tcgen05_dealloc(tensor)

    def slice(self, tensor: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> TMemoryTensor:
        return self._builder.tcgen05_slice(tensor, offsets, shape)

    def view(self, tensor: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> TMemoryTensor:
        return self._builder.tcgen05_view(tensor, dtype, shape)

    def relinquish_alloc_permit(self, cta_group: int) -> None:
        self._builder.tcgen05_relinquish_alloc_permit(cta_group)

    def load(self, tensor: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> RegisterTensor:
        return self._builder.tcgen05_load(tensor, offsets, shape)

    def store(self, tensor: TMemoryTensor, src: RegisterTensor, offsets: Sequence[int] = (0, 0)) -> None:
        return self._builder.tcgen05_store(tensor, src, offsets)

    def wait_load(self) -> None:
        self._builder.tcgen05_wait_load()

    def wait_store(self) -> None:
        self._builder.tcgen05_wait_store()

    def copy(self, src: SharedTensor, dst: TMemoryTensor) -> None:
        self._builder.tcgen05_copy(src, dst)

    def commit(self, mbarrier: Expr | RegisterTensor, cta_mask: Optional[int] = None) -> None:
        self._builder.tcgen05_commit(mbarrier, cta_mask)

    def mma(self, a: SharedTensor | TMemoryTensor, b: SharedTensor, d: TMemoryTensor) -> None:
        if isinstance(a, SharedTensor):
            self._builder.tcgen05_mma_ss(a, b, d)
        elif isinstance(a, TMemoryTensor):
            self._builder.tcgen05_mma_ts(a, b, d)
        else:
            raise InstructionError(f"Invalid type of a: {type(a)}, expected SharedTensor or TMemoryTensor")
