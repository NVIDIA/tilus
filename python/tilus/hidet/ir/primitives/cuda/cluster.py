# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import namedtuple
from typing import Union

from tilus.hidet.ir.dtypes import i32
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.ir.primitives.vars import lookup_primitive_variable, register_primitive_variable
from tilus.hidet.ir.type import DataType, FuncType, PointerType, VoidType, data_type
from tilus.hidet.utils.py import initialize

_cluster_fields = ["thread_rank", "block_rank", "dim_threads", "dim_blocks"]


@initialize()
def register_cuda_cluster_functions():
    for suffix in _cluster_fields:
        register_primitive_variable(name=f"cooperative_groups::this_cluster().{suffix}()", dtype=i32)

    register_primitive_function(
        name="this_cluster.sync",
        func_or_type=FuncType([], VoidType()),
        codegen_name="cooperative_groups::this_cluster().sync",
    )

    for dtype in ["int8", "uint8", "uint32", "uint64", "int32", "float16", "float32", "bool"]:
        dtype = data_type(dtype)

        register_primitive_function(
            name=f"this_cluster.map_shared_rank_{dtype}",
            func_or_type=FuncType([PointerType(dtype), i32], PointerType(dtype)),
            codegen_name="cooperative_groups::this_cluster().map_shared_rank",
        )


def cluster_sync():
    return call_primitive_func("this_cluster.sync", [])


def cluster_map_shared_rank(addr: Expr, rank: Union[Expr, int], dtype: Union[DataType, str]):
    func_name = f"this_cluster.map_shared_rank_{dtype}"
    return call_primitive_func(func_name, [addr, rank])


this_cluster = namedtuple("this_cluster", field_names=_cluster_fields + ["sync", "map_shared_rank"])(
    *[lookup_primitive_variable("cooperative_groups::this_cluster().{}()".format(field)) for field in _cluster_fields],
    cluster_sync,
    cluster_map_shared_rank,
)


# ---------------------------------------------------------------------------
# PTX-level cluster primitives (from extensions)
# ---------------------------------------------------------------------------
from typing import Literal, no_type_check

from tilus.hidet.ir.dtypes import int32 as int32_dt
from tilus.hidet.ir.primitives.cuda.funcs import call_cuda
from tilus.hidet.ir.stmt import asm
from tilus.hidet.lang import script


@initialize()
def register_cluster_instructions():
    """Register cluster-level PTX instruction primitives."""
    from tilus.hidet.lang import attrs

    @no_type_check
    @script
    def cuda_cluster_arrive_relaxed():
        attrs.func_name = "cuda_cluster_arrive_relaxed"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.arrive.relaxed.aligned;", outputs=[], inputs=[], is_volatile=True)

    @no_type_check
    @script
    def cuda_cluster_arrive():
        attrs.func_name = "cuda_cluster_arrive"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.arrive.aligned;", outputs=[], inputs=[], is_volatile=True)

    @no_type_check
    @script
    def cuda_cluster_wait():
        attrs.func_name = "cuda_cluster_wait"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.wait.aligned;", outputs=[], inputs=[], is_volatile=True)

    @no_type_check
    @script
    def cuda_cluster_sync_ptx():
        attrs.func_name = "cuda_cluster_sync"
        attrs.func_kind = "cuda_internal"
        asm(template="barrier.cluster.arrive.aligned;", outputs=[], inputs=[], is_volatile=True)
        asm(template="barrier.cluster.wait.aligned;", outputs=[], inputs=[], is_volatile=True)

    @no_type_check
    @script
    def cuda_cluster_grid_dim_x() -> int32_dt:
        attrs.func_name = "cuda_cluster_grid_dim_x"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%nclusterid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_grid_dim_y() -> int32_dt:
        attrs.func_name = "cuda_cluster_grid_dim_y"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%nclusterid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_grid_dim_z() -> int32_dt:
        attrs.func_name = "cuda_cluster_grid_dim_z"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%nclusterid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_id_in_grid_x() -> int32_dt:
        attrs.func_name = "cuda_cluster_id_in_grid_x"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%clusterid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_id_in_grid_y() -> int32_dt:
        attrs.func_name = "cuda_cluster_id_in_grid_y"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%clusterid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_id_in_grid_z() -> int32_dt:
        attrs.func_name = "cuda_cluster_id_in_grid_z"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%clusterid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_id_in_cluster_x() -> int32_dt:
        attrs.func_name = "cuda_block_id_in_cluster_x"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_ctaid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_id_in_cluster_y() -> int32_dt:
        attrs.func_name = "cuda_block_id_in_cluster_y"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_ctaid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_id_in_cluster_z() -> int32_dt:
        attrs.func_name = "cuda_block_id_in_cluster_z"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_ctaid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_shape_x() -> int32_dt:
        attrs.func_name = "cuda_cluster_shape_x"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_nctaid.x;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_blocks() -> int32_dt:
        attrs.func_name = "cuda_cluster_blocks"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_nctarank;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_shape_y() -> int32_dt:
        attrs.func_name = "cuda_cluster_shape_y"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_nctaid.y;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_cluster_shape_z() -> int32_dt:
        attrs.func_name = "cuda_cluster_shape_z"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_nctaid.z;", outputs=[ret], inputs=[])
        return ret

    @no_type_check
    @script
    def cuda_block_rank_in_cluster() -> int32_dt:
        attrs.func_name = "cuda_block_rank_in_cluster"
        attrs.func_kind = "cuda_internal"
        ret: int32_dt = 0
        asm(template="mov.s32 %0, %%cluster_ctarank;", outputs=[ret], inputs=[])
        return ret

    for func in [
        cuda_cluster_arrive_relaxed,
        cuda_cluster_arrive,
        cuda_cluster_blocks,
        cuda_cluster_wait,
        cuda_cluster_sync_ptx,
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
    return call_cuda("cluster_arrive_relaxed", args=[])


def cluster_arrive():
    return call_cuda("cluster_arrive", args=[])


def cluster_wait():
    return call_cuda("cluster_wait", args=[])


def cluster_sync_ptx():
    return call_cuda("cluster_sync", args=[])


def cluster_grid_dim(dim: Literal["x", "y", "z"]) -> Expr:
    return call_cuda(f"cluster_grid_dim_{dim}", args=[])


def cluster_id_in_grid(dim: Literal["x", "y", "z"]) -> Expr:
    return call_cuda(f"cluster_id_in_grid_{dim}", args=[])


def cluster_shape(dim: Literal["x", "y", "z"]) -> Expr:
    return call_cuda(f"cluster_shape_{dim}", args=[])


def block_id_in_cluster(dim: Literal["x", "y", "z"]) -> Expr:
    return call_cuda(f"block_id_in_cluster_{dim}", args=[])


def block_rank_in_cluster() -> Expr:
    return call_cuda("block_rank_in_cluster", args=[])


def cluster_blocks_ptx() -> Expr:
    return call_cuda("cluster_blocks", args=[])


# Alias for backward compatibility
cluster_blocks = cluster_blocks_ptx
