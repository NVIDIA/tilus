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
from tilus.hidet.ir.dtypes import uint32
from tilus.hidet.ir.expr import Expr, Var, as_expr
from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor
from tilus.lang.constructs.structs import Dim3

from .root import InstructionGroup


class BlockClusterInstructionGroup(InstructionGroup):
    def sync(self) -> None:
        from tilus.hidet.ir.primitives.cuda.cluster import cluster_sync

        self._builder.evaluate(pred=None, expr=cluster_sync())

    @property
    def blockIdx(self) -> Dim3:
        from tilus.hidet.ir.primitives.cuda.vars import clusterBlockIdx

        return Dim3(
            clusterBlockIdx.x,
            clusterBlockIdx.y,
            clusterBlockIdx.z,
        )

    @property
    def clusterDim(self) -> Dim3:
        from tilus.hidet.ir.primitives.cuda.vars import clusterDim

        return Dim3(
            clusterDim.x,
            clusterDim.y,
            clusterDim.z,
        )

    @property
    def blockRank(self) -> Var:
        from tilus.hidet.ir.primitives.cuda.vars import clusterBlockRank

        return clusterBlockRank

    def map_shared_addr(self, addr: RegisterTensor, target_rank: Expr | int) -> RegisterTensor:
        """Map shared memory address(es) to the corresponding address(es) in another CTA's shared memory.

        This instruction uses the PTX ``mapa.shared::cluster`` instruction to translate shared memory addresses
        from the current CTA's address space to another CTA's address space within the same cluster.

        Parameters
        ----------
        addr: RegisterTensor
            A register tensor of dtype uint32 containing shared memory address(es) to map.
        target_rank: Expr | int
            The rank of the target CTA in the cluster.

        Returns
        -------
        RegisterTensor
            A register tensor with the same shape and dtype as ``addr``, containing the mapped addresses.
        """
        if not isinstance(addr, RegisterTensor):
            raise InstructionError("addr must be a RegisterTensor, got {}".format(type(addr)))
        if addr.dtype != uint32:
            raise InstructionError("addr must have dtype uint32, got {}".format(addr.dtype))
        return self._builder.map_shared_addr(addr, target_rank=as_expr(target_rank))
