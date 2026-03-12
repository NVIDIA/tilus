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

from hidet.ir.dtypes import uint32, uint64
from hidet.ir.expr import Var

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.fence import fence_view_async
from tilus.extensions.hidet.ir.primitives.cuda.mapa import mapa_shared
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
)
from tilus.ir.instructions.cuda.fence import FenceViewAsync
from tilus.ir.instructions.cuda.mbarrier import (
    AllocBarrierInst,
    ArriveBarrierInst,
    ArriveExpectTxBarrierInst,
    ArriveExpectTxMulticastBarrierInst,
    ArriveExpectTxRemoteBarrierInst,
    WaitBarrierInst,
)
from tilus.target import nvgpu_sm80, nvgpu_sm90


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


@register_emitter(ArriveBarrierInst, target=nvgpu_sm80)
class ArriveBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveBarrierInst) -> None:
        self.append(mbarrier_arrive(inst.barrier, count=inst.count, sem=inst.sem, scope=inst.scope, space="cta"))


@register_emitter(ArriveExpectTxBarrierInst, target=nvgpu_sm90)
class ArriveExpectTxBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveExpectTxBarrierInst) -> None:
        self.append(
            mbarrier_arrive_expect_tx(
                mbarrier_addr=inst.barrier,
                transaction_bytes=inst.transaction_bytes,
                sem=inst.sem,
                scope=inst.scope,
                space="cta",
            )
        )


@register_emitter(ArriveExpectTxMulticastBarrierInst, target=nvgpu_sm90)
class ArriveExpectTxMulticastBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveExpectTxMulticastBarrierInst) -> None:
        if self.current_num_threads < 16:
            raise ValueError("Multicast mbarrier operations require at least 16 threads in the thread group.")

        with self.if_then((uint32(1) << self.current_thread) & uint32(inst.multicast)):
            self.append(
                mbarrier_arrive_expect_tx(
                    mbarrier_addr=mapa_shared(inst.barrier, cta_rank=self.current_thread),
                    transaction_bytes=inst.transaction_bytes,
                    sem=inst.sem,
                    scope=inst.scope,
                    space="cluster",
                )
            )


@register_emitter(ArriveExpectTxRemoteBarrierInst, target=nvgpu_sm90)
class ArriveExpectTxRemoteBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveExpectTxRemoteBarrierInst) -> None:
        self.append(
            mbarrier_arrive_expect_tx(
                mbarrier_addr=mapa_shared(inst.barrier, inst.target_rank),
                transaction_bytes=inst.transaction_bytes,
                sem=inst.sem,
                scope=inst.scope,
                space="cluster",
            )
        )


@register_emitter(WaitBarrierInst, target=nvgpu_sm90)
class WaitBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: WaitBarrierInst) -> None:
        self.append(mbarrier_wait(inst.barrier, inst.phase, sem=inst.sem, scope=inst.scope))


@register_emitter(FenceViewAsync, target=nvgpu_sm80)
class FenceViewAsyncEmitter(BaseInstEmitter):
    def emit(self, inst: FenceViewAsync) -> None:
        self.append(fence_view_async(scope=inst.space))
