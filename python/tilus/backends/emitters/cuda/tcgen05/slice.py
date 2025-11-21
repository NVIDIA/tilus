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

from hidet.ir.expr import Var

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    COLUMN_STRIDE,
)
from tilus.ir.instructions.cuda.tcgen05 import (
    Tcgen05SliceInst,
)
from tilus.ir.tensor import TMemoryTensor


@register_emitter(Tcgen05SliceInst)
class TMemorySliceEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05SliceInst) -> None:
        src: TMemoryTensor = inst.tmemory_input
        dst: TMemoryTensor = inst.tmemory_output

        src_addr: Var = self.tensor2var[src]
        dst_addr: Var = self.get_or_allocate_var(dst)

        column_offset = (
            sum(inst.offsets[dim] * src.layout.column_strides[dim] for dim in range(len(src.shape)))
            * src.dtype.nbits
            // 32
        )
        self.assign(dst_addr, src_addr + column_offset * COLUMN_STRIDE)
