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

from hidet.ir.primitives.cuda.barrier import fence_view_async_shared

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
)
from tilus.ir.instructions.cuda.mbarrier import (
    AllocBarrierInst,
    ArriveBarrierInst,
    ArriveExpectTxBarrierInst,
    WaitBarrierInst,
)
from tilus.target import nvgpu_sm80, nvgpu_sm90


@register_emitter(AllocBarrierInst, target=nvgpu_sm80)
class AllocBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocBarrierInst) -> None:
        out = inst.register_output
        out_var = self.get_or_allocate_var(out)

        counts = [c if c is not None else self.current_num_threads for c in inst.counts]
        barriers = self.contexts.barrier_alloc_ctx.allocate_barriers(counts=counts)

        for i in range(len(barriers)):
            self.buffer_store(out_var, indices=[i], value=barriers[i])
