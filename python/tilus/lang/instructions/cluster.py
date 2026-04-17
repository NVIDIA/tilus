# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    """Block cluster instructions for multi-CTA coordination on Hopper+ GPUs.

    A **cluster** is a group of thread blocks (CTAs) that can directly access each other's
    shared memory and synchronize collectively. Clusters are configured at launch time via
    ``self.attrs.cluster_blocks``.

    This instruction group provides:

    - **Synchronization**: ``sync()`` is a cluster-wide barrier — all threads across all CTAs
      in the cluster must arrive before any can proceed.
    - **Introspection**: ``blockIdx``, ``clusterDim``, and ``blockRank`` provide the current
      CTA's position and rank within the cluster.
    - **Cross-CTA addressing**: ``map_shared_addr()`` translates a shared memory address from
      the current CTA's address space to another CTA's, enabling direct remote shared memory
      access (e.g., signaling a peer CTA's mbarrier).
    """

    def sync(self) -> None:
        """Synchronize all thread blocks in the current cluster.

        All threads in all CTAs of the cluster must reach this barrier before any thread can
        proceed past it. This is the cluster-level equivalent of ``__syncthreads()``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``barrier.cluster.arrive`` / ``barrier.cluster.wait``
        """
        from tilus.hidet.ir.primitives.cuda.cluster import cluster_sync

        self._builder.evaluate(pred=None, expr=cluster_sync())

    @property
    def blockIdx(self) -> Dim3:
        """The block index within the cluster.

        Returns a ``Dim3`` with ``x``, ``y``, ``z`` components representing the position
        of the current block within the cluster grid.

        Notes
        -----
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        """
        from tilus.hidet.ir.primitives.cuda.vars import clusterBlockIdx

        return Dim3(
            clusterBlockIdx.x,
            clusterBlockIdx.y,
            clusterBlockIdx.z,
        )

    @property
    def clusterDim(self) -> Dim3:
        """The dimensions of the cluster.

        Returns a ``Dim3`` with ``x``, ``y``, ``z`` components representing the
        number of blocks in each dimension of the cluster.

        Notes
        -----
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        """
        from tilus.hidet.ir.primitives.cuda.vars import clusterDim

        return Dim3(
            clusterDim.x,
            clusterDim.y,
            clusterDim.z,
        )

    @property
    def blockRank(self) -> Var:
        """The linear rank of the current block within the cluster.

        Returns a scalar ``uint32`` value in the range ``[0, clusterSize)``.

        Notes
        -----
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        """
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``mapa.shared::cluster``
        """
        if not isinstance(addr, RegisterTensor):
            raise InstructionError("addr must be a RegisterTensor, got {}".format(type(addr)))
        if addr.dtype != uint32:
            raise InstructionError("addr must have dtype uint32, got {}".format(addr.dtype))
        return self._builder.map_shared_addr(addr, target_rank=as_expr(target_rank))
