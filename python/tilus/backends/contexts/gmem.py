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

from typing import Optional

from hidet.ir import StmtBuilder
from hidet.ir.dtypes import int32, uint8
from hidet.ir.expr import Expr, Var, cast
from hidet.ir.primitives.runtime import request_cuda_workspace

from tilus.backends.codegen import BaseEmitContext, FunctionCodegen, register_emit_context
from tilus.target import get_current_target


@register_emit_context
class GlobalMemoryAllocationContext(BaseEmitContext):
    _current: Optional[GlobalMemoryAllocationContext] = None

    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)

        self.gmem_base_ptr: Var = Var(hint="gmem", type=~uint8)
        self.gmem_allocated: Optional[Expr] = None

        self.gmem_clean_base_ptr: Var = Var(hint="gmem_clean", type=~uint8)
        self.gmem_clean_allocated: Optional[Expr] = None

    @staticmethod
    def current() -> GlobalMemoryAllocationContext:
        if GlobalMemoryAllocationContext._current is None:
            raise RuntimeError("No GlobalMemoryAllocationContext is currently active.")
        return GlobalMemoryAllocationContext._current

    def allocate_global_memory(self, nbytes: Expr | int, clean: bool) -> Expr:
        nbytes = (nbytes + 127) // 128 * 128  # align to 128 bytes
        if clean:
            if self.gmem_clean_allocated is None:
                self.gmem_clean_allocated = int32.zero
            ret = self.gmem_clean_base_ptr + self.gmem_clean_allocated
            self.gmem_clean_allocated = self.gmem_clean_allocated + nbytes
        else:
            if self.gmem_allocated is None:
                self.gmem_allocated = int32.zero
            ret = self.gmem_base_ptr + self.gmem_allocated
            self.gmem_allocated = self.gmem_allocated + nbytes
        return ret

    def finalize(self):
        target = get_current_target()
        request_workspace_functions = {
            "nvgpu": request_cuda_workspace,
        }
        if target.kind not in request_workspace_functions:
            raise NotImplementedError(f"Global memory allocation is not supported for target {target}")
        request_workspace = request_workspace_functions[target.kind]

        for allocated, base_ptr, clean in [
            (self.gmem_allocated, self.gmem_base_ptr, False),
            (self.gmem_clean_allocated, self.gmem_clean_base_ptr, True),
        ]:
            if allocated is None:
                continue
            sb = StmtBuilder()
            sb.declare(base_ptr, cast(request_workspace(nbytes=allocated, require_clean=clean), ~uint8))
            self.host_prepend(sb.finish())
            self.append_extra_param(base_ptr)
