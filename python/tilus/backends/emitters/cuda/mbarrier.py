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
from hidet.ir.dtypes import boolean, int32
from hidet.ir.primitives.cuda.barrier import fence_view_async_shared, mbarrier_arrive, mbarrier_init, mbarrier_wait

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import ArriveBarrierInst, ArriveRemoteBarrierInst, InitBarrierInst, WaitBarrierInst
from tilus.target import nvgpu_sm80, nvgpu_sm90


@register_emitter(InitBarrierInst, target=nvgpu_sm80)
class InitBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: InitBarrierInst) -> None:
        with self.if_then(self.current_worker == 0):
            count = inst.count if inst.count is not None else int32(self.current_num_workers)
            self.append(mbarrier_init(inst.barrier, count))
            self.append(fence_view_async_shared())


@register_emitter(ArriveBarrierInst, target=nvgpu_sm80)
class ArriveBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveBarrierInst) -> None:
        # self.append(printf("[%d, %d, %d][%d] arrive barrier %p\n", blockIdx.x, blockIdx.y, blockIdx.z, self.current_worker, inst.barrier))
        self.append(mbarrier_arrive(inst.barrier))


@register_emitter(ArriveRemoteBarrierInst, target=nvgpu_sm80)
class ArriveRemoteBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: ArriveRemoteBarrierInst) -> None:
        self.append(mbarrier_arrive(inst.barrier, inst.remote_block, pred=boolean.true))


@register_emitter(WaitBarrierInst, target=nvgpu_sm90)
class WaitBarrierInstEmitter(BaseInstEmitter):
    def emit(self, inst: WaitBarrierInst) -> None:
        # self.append(printf("[%d, %d, %d][%d] start waiting barrier %p phase %d\n", blockIdx.x, blockIdx.y, blockIdx.z, self.current_worker, inst.barrier, inst.phase))
        self.append(mbarrier_wait(inst.barrier, inst.phase))
        # self.append(printf("[%d, %d, %d][%d] end waiting barrier %p phase %d\n", blockIdx.x, blockIdx.y, blockIdx.z, self.current_worker, inst.barrier, inst.phase))
