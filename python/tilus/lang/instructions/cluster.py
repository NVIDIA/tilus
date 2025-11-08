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

    def block_id(self) -> Dim3:
        from tilus.extensions.hidet.ir.primitives.cuda.cluster import block_id_in_cluster

        return Dim3(
            self._builder.declare(type=int32, init=block_id_in_cluster("x"), hint="block_id_in_cluster_x"),
            self._builder.declare(type=int32, init=block_id_in_cluster("y"), hint="block_id_in_cluster_y"),
            self._builder.declare(type=int32, init=block_id_in_cluster("z"), hint="block_id_in_cluster_z"),
        )

    def shape(self) -> Dim3:
        from tilus.extensions.hidet.ir.primitives.cuda.cluster import cluster_shape

        return Dim3(
            self._builder.declare(type=int32, init=cluster_shape("x"), hint="cluster_dim_x"),
            self._builder.declare(type=int32, init=cluster_shape("y"), hint="cluster_dim_y"),
            self._builder.declare(type=int32, init=cluster_shape("z"), hint="cluster_dim_z"),
        )

    def block_rank(self) -> Var:
        from tilus.extensions.hidet.ir.primitives.cuda.cluster import block_rank_in_cluster

        return self._builder.declare(type=int32, init=block_rank_in_cluster(), hint="block_rank_in_cluster")
