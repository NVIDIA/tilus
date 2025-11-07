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
from hidet.ir.expr import logical_or

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.clc import (
    cluster_launch_control_query_response,
    cluster_launch_control_try_cancel,
)
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import (
    mbarrier_arrive_and_expect_tx_remote_shared,
    mbarrier_arrive_and_expect_tx_shared,
)
from tilus.ir.instructions.cuda.clc import ClusterLaunchControlQueryResponseInst, ClusterLaunchControlTryCancelInst
from tilus.ir.tensor import SharedTensor


@register_emitter(ClusterLaunchControlTryCancelInst)
class ClusterLaunchControlTryCancelEmitter(BaseInstEmitter):
    def emit(self, inst: ClusterLaunchControlTryCancelInst) -> None:
        response: SharedTensor = inst.shared_input
        if not inst.multicast:
            with self.if_then(logical_or(self.current_num_threads == 1, self.current_thread == 0)):
                self.append(mbarrier_arrive_and_expect_tx_shared(inst.mbarrier, transaction_bytes=16))
                self.append(
                    cluster_launch_control_try_cancel(
                        mbarrier=inst.mbarrier,
                        response=self.shared_tensor_shared_space_addr[response],
                        multicast=inst.multicast,
                    )
                )
        else:
            if self.current_num_threads < 32:
                raise ValueError(
                    "Cluster launch control multicast arrive requires at least 32 threads in the current thread group."
                )
            self.append(
                mbarrier_arrive_and_expect_tx_remote_shared(
                    inst.mbarrier,
                    transaction_bytes=16,
                    cta_id=self.current_thread,
                    pred=self.current_thread < self.blocks_per_cluster,
                )
            )
            with self.if_then(logical_or(self.current_num_threads == 1, self.current_thread == 0)):
                self.append(
                    cluster_launch_control_try_cancel(
                        mbarrier=inst.mbarrier,
                        response=self.shared_tensor_shared_space_addr[response],
                        multicast=inst.multicast,
                    )
                )


@register_emitter(ClusterLaunchControlQueryResponseInst)
class ClusterLaunchControlQueryResponseEmitter(BaseInstEmitter):
    def emit(self, inst: ClusterLaunchControlQueryResponseInst) -> None:
        response: SharedTensor = inst.shared_input
        predicated_cta = inst.register_output
        predicated_cta_var = self.get_or_allocate_var(predicated_cta)
        self.append(
            cluster_launch_control_query_response(
                response=self.shared_tensor_shared_space_addr[response], outputs=~predicated_cta_var[0]
            )
        )
