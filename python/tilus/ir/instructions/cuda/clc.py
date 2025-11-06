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

from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Expr

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor


@dataclass(frozen=True, eq=False)
class ClusterLaunchControlTryCancelInst(Instruction):
    mbarrier: Expr

    @staticmethod
    def create(response: SharedTensor, mbarrier: Expr) -> ClusterLaunchControlTryCancelInst:
        return ClusterLaunchControlTryCancelInst(output=None, inputs=(response,), mbarrier=mbarrier)


@dataclass(frozen=True, eq=False)
class ClusterLaunchControlIsCanceledInst(Instruction):
    @staticmethod
    def create(response: RegisterTensor) -> ClusterLaunchControlIsCanceledInst:
        predicate = RegisterTensor(dtype=boolean, shape=())
        return ClusterLaunchControlIsCanceledInst(output=predicate, inputs=(response,))


@dataclass(frozen=True, eq=False)
class ClusterLaunchControlGetFirstCtaInst(Instruction):
    @staticmethod
    def create(response: RegisterTensor) -> ClusterLaunchControlGetFirstCtaInst:
        cta_id = RegisterTensor(dtype=int32, shape=(3,))
        return ClusterLaunchControlGetFirstCtaInst(output=cta_id, inputs=(response,))
