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

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import AllocateGlobalInst, GlobalViewInst
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
