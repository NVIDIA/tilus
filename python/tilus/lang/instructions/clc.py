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
from hidet.ir.expr import Expr, Var

from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.lang.constructs.structs import Dim3

from .root import InstructionGroup


class ClusterLaunchControlInstructionGroup(InstructionGroup):
    def try_cancel(self, response: SharedTensor, mbarrier: Expr | RegisterTensor, multicast: Expr | bool) -> None:
        self._builder.cluster_launch_control_try_cancel(response, mbarrier, multicast)

    def query_response(self, response: SharedTensor) -> tuple[Var, Dim3]:
        ret = self._builder.cluster_launch_control_query_response(response)
        items = []
        for i in range(4):  # (is_canceled, first_cta_x, first_cta_y, first_cta_z)
            items.append(
                self._builder.tensor_item_value(
                    self._builder.slice_register(ret, offsets=[i], slice_dims=[], slice_shape=[])
                )
            )
        return (items[0], Dim3(items[1], items[2], items[3]))
