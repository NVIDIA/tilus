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
import contextlib
from typing import Optional, Sequence

from hidet.ir.expr import Expr
from hidet.ir.type import DataType

from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor
from tilus.utils import is_power_of_two, prod

from .root import InstructionGroup


class Tcgen05InstructionGroup(InstructionGroup):
    def alloc(
        self, dtype: DataType, shape: Sequence[int], cta_group: int = 1, init: Optional[Expr | float | int] = None
    ) -> TMemoryTensor:
        if cta_group not in [1, 2]:
            raise InstructionError("cta_group must be 1 or 2")
        if len(shape) < 2:
            raise InstructionError("shape must be a sequence of length 2 or more, got {}".format(shape))
        if shape[-2] not in (32, 64, 128):
            raise InstructionError("shape[-2] must be 32, 64, or 128, got {}".format(shape[-2]))
        if 128 % dtype.nbits != 0:
            raise InstructionError("dtype must be 1, 2, 4, 8, 16, 32, 64, or 128 bit, got {}".format(dtype))
        num_columns = prod(shape[:-2]) * shape[-1] * dtype.nbits // 32
        if not is_power_of_two(num_columns) or num_columns < 32 or num_columns > 512:
            raise InstructionError(
                f"The number of 32-bit columns requested: {num_columns}, "
                "it must be a power of two and in the range [32, 512]."
            )
        ret = self._builder.tcgen05_alloc(dtype, shape, cta_group)
        if init is not None:
            # check the thread group is valid to perform initialization
            tg_stack = self._builder.tg_stack
            thread_begin, thread_end = tg_stack.thread_begin[-1], tg_stack.thread_end[-1]
            if thread_begin % 128 != 0 or (thread_end - thread_begin) < 128:
                raise InstructionError(
                    "The thread group used to allocate with initialization must start at a multiple of 128 "
                    "and have at least 128 threads."
                )
            if thread_end - thread_begin == 128:
                ctx = contextlib.nullcontext()
            else:
                ctx = self._builder.thread_group(thread_begin=0, num_threads=128)
            with ctx:
                if len(shape) == 2:
                    self._builder.tcgen05_store(
                        ret,
                        src=self._builder.allocate_register(dtype=dtype, shape=shape, f_init=lambda _: dtype(init)),
                    )
                else:
                    with self._builder.for_grid(extents=shape[:-2]) as indices:
                        sub_shape = shape[-2:]
                        self._builder.tcgen05_store(
                            tmem=self._builder.tcgen05_slice(
                                ret, offsets=indices + [0, 0], slice_dims=[-2, -1], slice_shape=sub_shape
                            ),
                            src=self._builder.allocate_register(
                                dtype=dtype, shape=sub_shape, f_init=lambda _: dtype(init)
                            ),
                        )
                self._builder.tcgen05_wait_store()
            self._builder.syncthreads()
        return ret

    def dealloc(self, tensor: TMemoryTensor) -> None:
        self._builder.tcgen05_dealloc(tensor)

    def slice(
        self, tensor: TMemoryTensor, offsets: Sequence[Expr | int], dims: Sequence[int], shape: Sequence[int]
    ) -> TMemoryTensor:
        return self._builder.tcgen05_slice(tensor, offsets, dims, shape)

    def view(self, tensor: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> TMemoryTensor:
        return self._builder.tcgen05_view(tensor, dtype, shape)

    def relinquish_alloc_permit(self, cta_group: int) -> None:
        self._builder.tcgen05_relinquish_alloc_permit(cta_group)

    def load(self, tensor: TMemoryTensor) -> RegisterTensor:
        if len(tensor.shape) != 2:
            raise InstructionError("load requires a 2D tensor memory tensor, got shape {}".format(tensor.shape))
        return self._builder.tcgen05_load(tensor)

    def store(self, tensor: TMemoryTensor, src: RegisterTensor) -> None:
        if len(tensor.shape) != 2:
            raise InstructionError("store requires a 2D tensor memory tensor, got shape {}".format(tensor.shape))
        if len(src.shape) != 2:
            raise InstructionError("store requires a 2D register tensor, got shape {}".format(src.shape))
        return self._builder.tcgen05_store(tensor, src)

    def wait_load(self) -> None:
        self._builder.tcgen05_wait_load()

    def wait_store(self) -> None:
        self._builder.tcgen05_wait_store()

    def copy(self, src: SharedTensor, dst: TMemoryTensor) -> None:
        if len(src.shape) != 2:
            raise InstructionError("copy requires a 2D shared tensor, got shape {}".format(src.shape))
        if len(dst.shape) != 2:
            raise InstructionError("copy requires a 2D tensor memory tensor, got shape {}".format(dst.shape))
        self._builder.tcgen05_copy(src, dst)

    def commit(self, mbarrier: Expr | RegisterTensor, cta_mask: Optional[int] = None) -> None:
        self._builder.tcgen05_commit(mbarrier, cta_mask)

    def mma(self, a: SharedTensor | TMemoryTensor, b: SharedTensor, d: TMemoryTensor) -> None:
        if isinstance(a, SharedTensor):
            if len(a.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(a.shape))
            if len(b.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(b.shape))
            if len(d.shape) != 2:
                raise InstructionError(
                    "mma requires a 2D tensor memory tensor for output, got shape {}".format(d.shape)
                )
            self._builder.tcgen05_mma_ss(a, b, d)
        elif isinstance(a, TMemoryTensor):
            if len(a.shape) != 2:
                raise InstructionError("mma requires a 2D tensor memory tensor, got shape {}".format(a.shape))
            if len(b.shape) != 2:
                raise InstructionError("mma requires a 2D shared tensor, got shape {}".format(b.shape))
            if len(d.shape) != 2:
                raise InstructionError(
                    "mma requires a 2D tensor memory tensor for output, got shape {}".format(d.shape)
                )
            self._builder.tcgen05_mma_ts(a, b, d)
        else:
            raise InstructionError(f"Invalid type of a: {type(a)}, expected SharedTensor or TMemoryTensor")
