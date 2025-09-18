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
from hidet.ir import logical_or
from hidet.ir.dtypes import int32, uint32
from hidet.ir.expr import cast

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.backends.contexts import SharedMemoryAllocationContext, Tcgen05EmitContext
from tilus.extensions.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_relinquish_alloc_permit,
)
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05AllocInst, Tcgen05DeallocInst, Tcgen05RelinquishAllocPermitInst
from tilus.target import nvgpu_sm100


@register_emitter(Tcgen05AllocInst, target=nvgpu_sm100)
class Tcgen05AllocEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05AllocInst) -> None:
        if self.current_num_threads < 32:
            raise ValueError("tcgen05_alloc requires at least 32 threads in the current thread group")

        # set the cta group in the tcgen05 context
        tcgen05_ctx = Tcgen05EmitContext.current()
        tcgen05_ctx.set_cta_group(inst.cta_group)

        # allocate a workspace in shared memory to hold the tensor memory handle
        smem_ctx = SharedMemoryAllocationContext.current()
        smem_ptr = smem_ctx.request_shared_workspace(nbytes=4)

        # call tcgen05_alloc
        with self.if_then(logical_or(self.current_num_threads == 32, self.current_thread // 32 == 0)):
            smem_addr = self.declare_var("smem_addr", tp=uint32, init=cvta_generic_to_shared(smem_ptr))
            self.append(
                tcgen05_alloc(
                    dst=smem_addr,
                    num_columns=uint32(inst.output.as_tmemory_tensor().shape[1]),
                    cta_group=inst.cta_group,
                )
            )

        # let other warps in the thread group wait until the first warp finishes
        with self.if_then(self.current_num_threads > 32):
            self.sync()

        # load the tensor memory handle from shared memory and store it to the register variable
        tmem_var = self.get_or_allocate_var(tensor=inst.output)
        self.assign(tmem_var, cast(smem_ptr, ~int32)[0])
        self.sync()


@register_emitter(Tcgen05DeallocInst, target=nvgpu_sm100)
class Tcgen05DeallocEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05DeallocInst) -> None:
        tmem_var = self.get_or_allocate_var(tensor=inst.inputs[0].as_tmemory_tensor())
        num_columns = inst.inputs[0].as_tmemory_tensor().shape[1]
        tcgen05_ctx = Tcgen05EmitContext.current()
        self.append(tcgen05_dealloc(taddr=tmem_var, num_columns=uint32(num_columns), cta_group=tcgen05_ctx.cta_group))


@register_emitter(Tcgen05RelinquishAllocPermitInst, target=nvgpu_sm100)
class Tcgen05RelinquishAllocPermitEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05RelinquishAllocPermitInst) -> None:
        self.append(tcgen05_relinquish_alloc_permit(inst.cta_group))
