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
from hidet.ir.dtypes import int32
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.type import tensor_pointer_type

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.backends.contexts import SharedMemoryAllocationContext
from tilus.ir.instructions import AllocateSharedInst, FreeSharedInst, PermuteSharedInst, SliceSharedInst
from tilus.ir.tensor import SharedTensor


@register_emitter(AllocateSharedInst)
class AllocateSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateSharedInst) -> None:
        tensor: SharedTensor = inst.shared_output

        ctx: SharedMemoryAllocationContext = SharedMemoryAllocationContext.current()

        allocator_addr = ctx.allocate_shared_tensor(tensor, nbytes=tensor.nbytes)
        self.tensor2var[tensor] = self.declare_var(
            name="shared",
            tp=tensor_pointer_type(dtype=tensor.dtype, shape=[tensor.size]),
            init=dynamic_shared_memory(byte_offset=allocator_addr, dtype=tensor.dtype),
        )
        shared_space_addr = cvta_generic_to_shared(self.tensor2var[tensor])
        self.shared_tensor_shared_space_addr[tensor] = self.declare_var(
            name="shared_addr", tp=int32, init=shared_space_addr
        )


@register_emitter(FreeSharedInst)
class FreeSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: FreeSharedInst) -> None:
        tensor: SharedTensor = inst.inputs[0].as_shared_tensor()

        ctx: SharedMemoryAllocationContext = SharedMemoryAllocationContext.current()
        ctx.free_shared_tensor(tensor)

        del self.tensor2var[tensor]
        del self.shared_tensor_shared_space_addr[tensor]


@register_emitter(SliceSharedInst)
class SliceSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: SliceSharedInst) -> None:
        shared_input: SharedTensor = inst.shared_input
        shared_output: SharedTensor = inst.shared_output
        slice_offset = shared_input.layout(*inst.offsets)
        output_var = self.get_or_allocate_var(shared_output)
        self.assign(output_var, ~self.tensor2var[shared_input][slice_offset])
        self.shared_tensor_shared_space_addr[shared_output] = self.declare_var(
            "shared_addr",
            tp=int32,
            init=self.shared_tensor_shared_space_addr[shared_input] + slice_offset * shared_input.dtype.nbytes,
        )


@register_emitter(PermuteSharedInst)
class PermuteSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: PermuteSharedInst) -> None:
        shared_input: SharedTensor = inst.shared_input
        shared_output: SharedTensor = inst.shared_output

        output_var = self.get_or_allocate_var(shared_output)

        self.assign(output_var, self.tensor2var[shared_input])
        self.shared_tensor_shared_space_addr[shared_output] = self.declare_var(
            "shared_addr",
            tp=int32,
            init=self.shared_tensor_shared_space_addr[shared_input],
        )
