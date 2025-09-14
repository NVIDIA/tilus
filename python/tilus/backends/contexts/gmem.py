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
from typing import Optional

from hidet.ir.dtypes import int32, uint8
from hidet.ir.expr import Expr, Var

from tilus.backends.codegen import BaseEmitContext, FunctionCodegen


class GlobalMemoryAllocationContext(BaseEmitContext):
    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)

        self.gmem_base_ptr: Var = Var(hint='gmem', type=~uint8)
        self.gmem_allocated: Optional[Expr] = None

        self.gmem_clean_base_ptr: Var = Var(hint='gmem_clean', type=~uint8)
        self.gmem_clean_allocated: Optional[Expr] = None

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
        if self.gmem_allocated is not None:
            pass

        if self.gmem_clean_allocated is not None:
            pass
