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
from __future__ import annotations

from dataclasses import dataclass

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor


@dataclass(frozen=True, eq=False)
class ClusterLaunchControlTryCancelInst(Instruction):
    mbarrier: Expr
    multicast: Expr

    @staticmethod
    def create(response: SharedTensor, mbarrier: Expr, multicast: Expr) -> ClusterLaunchControlTryCancelInst:
        return ClusterLaunchControlTryCancelInst(
            output=None, inputs=(response,), mbarrier=mbarrier, multicast=multicast
        )


@dataclass(frozen=True, eq=False)
class ClusterLaunchControlQueryResponseInst(Instruction):
    @staticmethod
    def create(response: SharedTensor) -> ClusterLaunchControlQueryResponseInst:
        cta_pred = RegisterTensor(dtype=int32, shape=(4,))  # (is_valid, ctaid_x, ctaid_y, ctaid_z)
        return ClusterLaunchControlQueryResponseInst(output=cta_pred, inputs=(response,))
