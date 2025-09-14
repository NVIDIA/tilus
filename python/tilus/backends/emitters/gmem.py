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
from hidet.ir.expr import Expr

from tilus.backends.codegen import BaseInstEmitter, BaseEmitContext, FunctionCodegen, register_emitter
from tilus.ir import GlobalTensor
from tilus.ir.instructions import AllocateGlobalInst, GlobalSliceInst, GlobalViewInst
from tilus.utils import cdiv


@register_emitter(GlobalViewInst)
class GlobalViewInstEmitter(BaseInstEmitter):
    def emit(self, inst: GlobalViewInst) -> None:
        self.assign(self.get_or_allocate_var(inst.global_output), inst.ptr)


@register_emitter(AllocateGlobalInst)
class AllocateGlobalInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateGlobalInst) -> None:
        tensor = inst.global_output
        ptr: Expr = self.codegen.allocate_global_memory(
            nbytes=cdiv(tensor.layout.size * tensor.dtype.nbits * 8, 8), clean=inst.require_clean
        )
        var = self.get_or_allocate_var(tensor)
        self.assign(var, ptr)


@register_emitter(GlobalSliceInst)
class GlobalSliceInstEmitter(BaseInstEmitter):
    def emit(self, inst: GlobalSliceInst) -> None:
        input_tensor: GlobalTensor = inst.global_input
        output_tensor: GlobalTensor = inst.global_output
        slice_offset = input_tensor.layout(*inst.offsets)
        output_var = self.get_or_allocate_var(output_tensor)
        self.assign(output_var, ~self.tensor2var[input_tensor][slice_offset])
