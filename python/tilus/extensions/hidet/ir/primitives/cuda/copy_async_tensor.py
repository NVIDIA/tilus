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
from typing import Optional, Sequence

from hidet.ir.type import OpaqueType
from hidet.ir.dtypes import int32, uint16, uint32, uint64
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.type import void_p
from hidet.utils import initialize
from tilus.extensions.hidet.ir.primitives.utils import register_primitive_function_decorator
from tilus.extensions.hidet.ir.primitives.cuda.tensor_map import CUtensorMapType




def resolve_cp_async_bulk_global_to_shared(dim: int, cache_hint: bool) -> str:
    func_name = "cp_async_bulk_tensor_{}d_shared_global{}".format(dim, "_cache_hint" if cache_hint else "")
    return func_name


def resolve_cp_async_bulk_global_to_cluster_shared(dim: int, cache_hint: bool) -> str:
    func_name = "cp_async_bulk_tensor_{}d_cluster_shared_global{}".format(dim, "_cache_hint" if cache_hint else "")
    return func_name

def resolve_cp_async_bulk_shared_to_global(dim: int, cache_hint: bool) -> str:
    func_name = "cp_async_bulk_tensor_{}d_global_shared{}".format(dim, "_cache_hint" if cache_hint else "")
    return func_name


@initialize()
def register_copy_async_tensor():
    from hidet.lang import asm, attrs, script, meta

    for dim in [1, 2, 3, 4, 5]:
        for cta_group in [1, 2]:
            for has_cache_hint in [False, True]:
                func_name = resolve_cp_async_bulk_global_to_shared(dim=dim, cache_hint=False)
                inst = "cp.async.bulk.tensor.{}d.shared::global.tile.mbarrier::complete_tx::bytes.cta_group::{}{}".format(
                    dim, cta_group, ".L2::cache_hint" if has_cache_hint else ""
                )
                if has_cache_hint:
                    operands = "[%0], [%1, {{{}}}], [%2], %3".format(', '.join(['%{}'.format(i + 4) for i in range(dim)]))
                else:
                    operands = "[%0], [%1, {{{}}}], [%2]".format(', '.join(['%{}'.format(i + 3) for i in range(dim)]))
                template = inst + ' ' + operands + ';'
                coords_type = meta.types([int32 for _ in range(dim)])
                cache_hint_type = meta.types([uint64] if has_cache_hint else [])
                @register_primitive_function_decorator
                @script
                def cp_async_bulk(
                        dst: uint32,
                        src: void_p,
                        tensor_map: CUtensorMapType,
                        coords: coords_type,
                        mbarrier: uint32,
                        cache_hint: cache_hint_type
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = "cuda_internal"
                    asm(
                        template=template,
                        inputs=[dst, src, tensor_map, mbarrier, *cache_hint, *coords],
                        is_volatile=True,
                        memory_fence=True
                    )


def cp_async_tensor_global_to_shared(
    dst: Expr, src: Expr, tensor_map: Expr, coords: Sequence[Expr], mbarrier: Expr, cache_policy: Optional[Expr] = None
) -> Expr:
    """Perform a bulk copy from global memory to shared memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in shared memory. It should be an address with shared memory space with type uint32.
    src: Expr
        The source address in global memory. It should be an address with global memory space with type void_p.
    tensor_map: Expr
        The tensor map that describes how the data is laid out in the global memory. It should be of type
        CUtensorMapType.
    coords: Sequence[Expr]
        The coordinates of the tile to be copied.
    mbarrier: Expr
        The mbarrier to be used for synchronization. It should be an address with shared memory space with type uint32
        that has been initialized by `mbarrier_init`.
    cache_policy: Expr, optional
        The cache policy to be used.
    Returns
    -------
    ret: Expr
        A function call expression.
    """
    assert cache_policy is None, "cache_policy is not supported yet"
    func_name = resolve_cp_async_bulk_global_to_shared(dim=len(coords), cache_hint=cache_policy is not None)
    optional_cache_policy = [cache_policy] if cache_policy is not None else []
    return call_cuda(func_name, args=[dst, src, tensor_map, *coords, mbarrier, *optional_cache_policy])
