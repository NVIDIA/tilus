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
import typing

from hidet.ir.dtypes import boolean, int32, uint32, uint64
from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.barrier import fence_view_async_shared
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import (
    mbarrier_arrive_cluster_shared,
    mbarrier_arrive_cta_shared,
    mbarrier_init_shared,
    mbarrier_wait_shared,
)
from tilus.ir.instructions.cuda.mbarrier import (
    AllocBarrierInst,
    ArriveBarrierInst,
    ArriveRemoteBarrierInst,
    FenceProxyCopyAsync,
    WaitBarrierInst,
)
from tilus.ir.tensor import SharedTensor
from tilus.target import nvgpu_sm80, nvgpu_sm90


@register_emitter(AllocBarrierInst, target=nvgpu_sm80)
class AllocBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocBarrierInst) -> None:
        out = inst.register_output
        out_var = self.get_or_allocate_var(out)

        smem_ctx = self.contexts.smem_alloc_ctx
        smem_tensor = SharedTensor.create(dtype=uint64, shape=[len(inst.counts)])
        mbarriers_offset = smem_ctx.allocate_shared_tensor(smem_tensor, nbytes=uint64.nbytes * len(inst.counts))
        mbarriers_addr = cvta_generic_to_shared(dynamic_shared_memory(byte_offset=mbarriers_offset, dtype=uint64))
        mbarriers_var = self.declare_var(name="mbarriers", tp=uint32, init=mbarriers_addr)

        with self.for_range(extent=len(inst.counts)) as i:
            self.buffer_store(out_var, indices=[i], value=mbarriers_var + uint64.nbytes * i)

        with self.if_then(self.current_thread == 0):
            for i in range(len(inst.counts)):
                count: Expr
                if inst.counts[i] is None:
                    count = typing.cast(Expr, int32(self.current_num_threads))
                else:
                    count = typing.cast(Expr, inst.counts[i])
                self.append(mbarrier_init_shared(out_var[i], arrive_count=count))
        self.append(fence_view_async_shared())
        self.sync()


# @register_emitter(InitBarrierInst, target=nvgpu_sm80)
# class InitBarrierInstEmitter(BaseInstEmitter):
#     def emit(self, inst: InitBarrierInst) -> None:
#         with self.if_then(self.current_thread == 0):
#             count = inst.count if inst.count is not None else int32(self.current_num_threads)
#             self.append(mbarrier_init_shared(inst.barrier, count))
#             self.append(fence_view_async_shared())


@register_emitter(ArriveBarrierInst, target=nvgpu_sm80)
class ArriveBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveBarrierInst) -> None:
        self.append(mbarrier_arrive_cta_shared(inst.barrier))


@register_emitter(ArriveRemoteBarrierInst, target=nvgpu_sm80)
class ArriveRemoteBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveRemoteBarrierInst) -> None:
        self.append(mbarrier_arrive_cluster_shared(inst.barrier, inst.remote_block, pred=boolean.true))


@register_emitter(WaitBarrierInst, target=nvgpu_sm90)
class WaitBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: WaitBarrierInst) -> None:
        self.append(mbarrier_wait_shared(inst.barrier, inst.phase))


@register_emitter(FenceProxyCopyAsync, target=nvgpu_sm90)
class FenceProxyCopyAsyncEmitter(BaseInstEmitter):
    def emit(self, inst: FenceProxyCopyAsync) -> None:
        self.append(fence_view_async_shared())
