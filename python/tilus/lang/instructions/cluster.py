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
from hidet.ir.dtypes import int32
from hidet.ir.expr import Var

from tilus.lang.constructs.structs import Dim3

from .root import InstructionGroup


class BlockClusterInstructionGroup(InstructionGroup):
    def sync(self) -> None:
        from tilus.extensions.hidet.ir.primitives.cuda.cluster import cluster_sync

        self._builder.evaluate(pred=None, expr=cluster_sync())

    @property
    def blockIdx(self) -> Dim3:
        from tilus.extensions.hidet.ir.primitives.cuda.vars import clusterBlockIdx

        return Dim3(
            clusterBlockIdx.x,
            clusterBlockIdx.y,
            clusterBlockIdx.z,
        )

    @property
    def clusterDim(self) -> Dim3:
        from tilus.extensions.hidet.ir.primitives.cuda.vars import clusterDim

        return Dim3(
            clusterDim.x,
            clusterDim.y,
            clusterDim.z,
        )


    @property
    def blockRank(self) -> Var:
        from tilus.extensions.hidet.ir.primitives.cuda.vars import clusterBlockRank

        return clusterBlockRank
