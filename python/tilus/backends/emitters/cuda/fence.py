# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.dtypes import uint32, uint64
from tilus.hidet.ir.expr import Var
from tilus.ir.instructions.cuda.mbarrier import (
    AllocBarrierInst,
)
from tilus.target import nvgpu_sm80


@register_emitter(AllocBarrierInst, target=nvgpu_sm80)
class AllocBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocBarrierInst) -> None:
        out = inst.register_output
        out_var = self.get_or_allocate_var(out)

        counts = [c if c is not None else self.current_num_threads for c in inst.counts]
        base_addr, barrier_vars = self.contexts.barrier_alloc_ctx.allocate_barriers(counts=counts)

        for i in range(len(barrier_vars)):
            self.buffer_store(out_var, indices=[i], value=barrier_vars[i])

        # Register as CTA-invariant tensor: value(i) = base_addr + i * uint64.nbytes
        axis = Var("i", type=uint32)
        expr = base_addr + axis * uint32(uint64.nbytes)
        self.contexts.const_reg_ctx.register(out, axes=[axis], expr=expr)
