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
"""CUDA cluster-level primitives for SM90+ architectures.

This module provides low-level PTX primitives for cluster synchronization
and cluster/block identification within a cluster programming model.
"""
from typing import Literal, no_type_check

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.stmt import asm
from hidet.lang import script
from hidet.utils import initialize


@initialize()
def register_cluster_instructions():
    """Register cluster-level PTX instruction primitives."""
    from hidet.lang import attrs

    @no_type_check
    @script
    def cuda_cluster_arrive_relaxed():
        """Relaxed cluster barrier arrival (non-blocking)."""
        attrs.func_name = "cuda_cluster_arrive_relaxed"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.arrive.relaxed.aligned;", outputs=[], inputs=[])

    @no_type_check
    @script
    def cuda_cluster_arrive():
        """Cluster barrier arrival."""
        attrs.func_name = "cuda_cluster_arrive"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.arrive.aligned;", outputs=[], inputs=[])

    @no_type_check
    @script
    def cuda_cluster_wait():
        """Wait for cluster barrier completion."""
        attrs.func_name = "cuda_cluster_wait"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.wait.aligned;", outputs=[], inputs=[])

    @no_type_check
    @script
    def cuda_cluster_sync():
        """Synchronize all blocks in cluster (arrive + wait)."""
        attrs.func_name = "cuda_cluster_sync"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.arrive.aligned;", outputs=[], inputs=[])
        asm(template="barrier.cluster.wait.aligned;", outputs=[], inputs=[])

    @no_type_check
    @script
    def cuda_cluster_grid_dim_x() -> int32:
        """Get number of clusters in grid (x dimension)."""
        attrs.func_name = "cuda_cluster_grid_dim_x"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%nclusterid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_grid_dim_y() -> int32:
        """Get number of clusters in grid (y dimension)."""
        attrs.func_name = "cuda_cluster_grid_dim_y"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%nclusterid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_grid_dim_z() -> int32:
        """Get number of clusters in grid (z dimension)."""
        attrs.func_name = "cuda_cluster_grid_dim_z"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%nclusterid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_id_in_grid_x() -> int32:
        """Get cluster ID in grid (x dimension)."""
        attrs.func_name = "cuda_cluster_id_in_grid_x"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%clusterid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_id_in_grid_y() -> int32:
        """Get cluster ID in grid (y dimension)."""
        attrs.func_name = "cuda_cluster_id_in_grid_y"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%clusterid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_id_in_grid_z() -> int32:
        """Get cluster ID in grid (z dimension)."""
        attrs.func_name = "cuda_cluster_id_in_grid_z"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%clusterid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_id_in_cluster_x() -> int32:
        """Get block ID within cluster (x dimension)."""
        attrs.func_name = "cuda_block_id_in_cluster_x"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_ctaid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_id_in_cluster_y() -> int32:
        """Get block ID within cluster (y dimension)."""
        attrs.func_name = "cuda_block_id_in_cluster_y"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_ctaid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_id_in_cluster_z() -> int32:
        """Get block ID within cluster (z dimension)."""
        attrs.func_name = "cuda_block_id_in_cluster_z"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_ctaid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_shape_x() -> int32:
        """Get cluster shape (x dimension)."""
        attrs.func_name = "cuda_cluster_shape_x"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_nctaid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_blocks() -> int32:
        """Get number of blocks in the cluster (1D)."""
        attrs.func_name = "cuda_cluster_blocks"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_nctarank;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_shape_y() -> int32:
        """Get cluster shape (y dimension)."""
        attrs.func_name = "cuda_cluster_shape_y"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_nctaid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_shape_z() -> int32:
        """Get cluster shape (z dimension)."""
        attrs.func_name = "cuda_cluster_shape_z"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_nctaid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_rank_in_cluster() -> int32:
        """Get 1D block rank within cluster."""
        attrs.func_name = "cuda_block_rank_in_cluster"
        attrs.func_kind = "cuda_internal"
        ret: int32 = 0
        asm(template="mov.s32 %0, %%cluster_ctarank;", outputs=[ret], inputs=[])
        return ret

    # Register all functions
    for func in [
        cuda_cluster_arrive_relaxed,
        cuda_cluster_arrive,
        cuda_cluster_blocks,
        cuda_cluster_wait,
        cuda_cluster_sync,
        cuda_cluster_grid_dim_x,
        cuda_cluster_grid_dim_y,
        cuda_cluster_grid_dim_z,
        cuda_cluster_id_in_grid_x,
        cuda_cluster_id_in_grid_y,
        cuda_cluster_id_in_grid_z,
        cuda_block_id_in_cluster_x,
        cuda_block_id_in_cluster_y,
        cuda_block_id_in_cluster_z,
        cuda_cluster_shape_x,
        cuda_cluster_shape_y,
        cuda_cluster_shape_z,
        cuda_block_rank_in_cluster,
    ]:
        register_primitive_function(name=func.name, func_or_type=func)


def cluster_arrive_relaxed():
    """
    Perform a relaxed cluster barrier arrival (non-blocking).

    This instruction signals arrival at the cluster barrier without enforcing memory ordering.
    Equivalent to CuTe's cluster_arrive_relaxed(). Requires SM90+ architecture.

    See Also
    --------
    cluster_arrive : Cluster barrier arrival with memory ordering
    cluster_wait : Wait for cluster barrier completion
    """
    return call_cuda("cluster_arrive_relaxed", args=[])


def cluster_arrive():
    """
    Perform a cluster barrier arrival.

    This instruction signals arrival at the cluster barrier with proper memory ordering.
    Equivalent to CuTe's cluster_arrive(). Requires SM90+ architecture.

    See Also
    --------
    cluster_wait : Wait for cluster barrier completion
    cluster_sync : Combined arrive and wait
    """
    return call_cuda("cluster_arrive", args=[])


def cluster_wait():
    """
    Wait for cluster barrier completion.

    This instruction blocks until all blocks in the cluster have arrived at the barrier.
    Equivalent to CuTe's cluster_wait(). Requires SM90+ architecture.

    See Also
    --------
    cluster_arrive : Signal arrival at cluster barrier
    cluster_sync : Combined arrive and wait
    """
    return call_cuda("cluster_wait", args=[])


def cluster_sync():
    """
    Synchronize all blocks in the cluster.

    This function performs a full cluster synchronization by calling cluster_arrive()
    followed by cluster_wait(). Equivalent to CuTe's cluster_sync().
    Requires SM90+ architecture.

    See Also
    --------
    cluster_arrive : Signal arrival at cluster barrier
    cluster_wait : Wait for cluster barrier completion
    """
    return call_cuda("cluster_sync", args=[])


def cluster_grid_dim(dim: Literal['x', 'y', 'z']) -> Expr:
    """
    Get the number of clusters in the grid along the specified dimension.

    Equivalent to CuTe's cluster_grid_dims()[dim]. Requires SM90+ architecture.

    Parameters
    ----------
    dim : {'x', 'y', 'z'}
        The dimension to query.

    Returns
    -------
    ret : Expr
        The number of clusters in the grid along the specified dimension.
    """
    return call_cuda(f"cluster_grid_dim_{dim}", args=[])


def cluster_id_in_grid(dim: Literal['x', 'y', 'z']) -> Expr:
    """
    Get the cluster ID in the grid along the specified dimension.

    Equivalent to CuTe's cluster_id_in_grid()[dim]. Requires SM90+ architecture.

    Parameters
    ----------
    dim : {'x', 'y', 'z'}
        The dimension to query.

    Returns
    -------
    ret : Expr
        The cluster ID in the grid along the specified dimension.
    """
    return call_cuda(f"cluster_id_in_grid_{dim}", args=[])


def cluster_shape(dim: Literal['x', 'y', 'z']) -> Expr:
    """
    Get the cluster shape along the specified dimension.

    Returns the number of blocks in the cluster along the specified dimension.
    Equivalent to CuTe's cluster_shape()[dim]. Requires SM90+ architecture.

    Parameters
    ----------
    dim : {'x', 'y', 'z'}
        The dimension to query.

    Returns
    -------
    ret : Expr
        The number of blocks in the cluster along the specified dimension.
    """
    return call_cuda(f"cluster_shape_{dim}", args=[])


def block_id_in_cluster(dim: Literal['x', 'y', 'z']) -> Expr:
    """
    Get the block ID within the cluster along the specified dimension.

    Equivalent to CuTe's block_id_in_cluster()[dim]. Requires SM90+ architecture.

    Parameters
    ----------
    dim : {'x', 'y', 'z'}
        The dimension to query.

    Returns
    -------
    ret : Expr
        The block ID within the cluster along the specified dimension.
    """
    return call_cuda(f"block_id_in_cluster_{dim}", args=[])


def block_rank_in_cluster() -> Expr:
    """
    Get the 1D block rank within the cluster.

    Returns a linearized rank for the current block within the cluster.
    Equivalent to CuTe's block_rank_in_cluster(). Requires SM90+ architecture.

    Returns
    -------
    ret : Expr
        The 1D block rank within the cluster.
    """
    return call_cuda("block_rank_in_cluster", args=[])

def cluster_blocks() -> Expr:
    """
    Get the number of blocks in the cluster (1D).

    Returns the total number of blocks in the cluster as a single integer.
    Equivalent to CuTe's cluster_blocks(). Requires SM90+ architecture.

    Returns
    -------
    ret : Expr
        The number of blocks in the cluster (1D).
    """
    return call_cuda("cluster_blocks", args=[])
