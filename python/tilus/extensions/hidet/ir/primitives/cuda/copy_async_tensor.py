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
from typing import Optional, Sequence, no_type_check

from hidet.ir.dtypes import int32, uint16, uint32, uint64
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func
from hidet.utils import initialize

from tilus.extensions.hidet.ir.primitives.cuda.tensor_map import CUTensorMapPointerType
from tilus.extensions.hidet.ir.primitives.utils import register_primitive_function_decorator


def resolve_cp_async_bulk_tensor_global_to_shared(dim: int, cta_group: Optional[int], cache_hint: bool) -> str:
    cta_group_item = "" if cta_group is None else "_cta_group_{}".format(cta_group)
    cache_hint_item = "" if not cache_hint else "_cache_hint"
    func_name = "cuda_cp_async_bulk_tensor_{}d_shared_global{}{}".format(dim, cta_group_item, cache_hint_item)
    return func_name


def resolve_cp_async_bulk_tensor_global_to_cluster_shared(
    dim: int,
    multicast: bool,
    cta_group: Optional[int],
    cache_hint: bool,
) -> str:
    multicast_str = "_multicast" if multicast else ""
    cta_group_str = "" if cta_group is None else "_cta_group_{}".format(cta_group)
    cache_hint_item = "" if not cache_hint else "_cache_hint"
    func_name = "cuda_cp_async_bulk_tensor_{}d_cluster_shared_global{}{}{}".format(
        dim, multicast_str, cta_group_str, cache_hint_item
    )
    return func_name


def resolve_cp_async_bulk_tensor_shared_to_global(dim: int, cache_hint: bool) -> str:
    cache_hint_item = "" if not cache_hint else "_cache_hint"
    func_name = "cuda_cp_async_bulk_tensor_{}d_global_shared{}".format(dim, cache_hint_item)
    return func_name


@initialize()
def register_copy_async_tensor():
    from hidet.lang import asm, attrs, meta

    from tilus.extensions.hidet.lang import script

    for dim in [1, 2, 3, 4, 5]:
        for cta_group in [None, 1, 2]:
            for has_cache_hint in [False, True]:
                func_name = resolve_cp_async_bulk_tensor_global_to_shared(
                    dim=dim, cta_group=cta_group, cache_hint=has_cache_hint
                )
                cta_group_item = "" if cta_group is None else ".cta_group::{}".format(cta_group)
                cache_hint_item = "" if not has_cache_hint else "::L2::cache_hint"
                inst = "cp.async.bulk.tensor.{}d.shared::cta.global.tile.mbarrier::complete_tx::bytes{}{}".format(
                    dim, cta_group_item, cache_hint_item
                )
                if has_cache_hint:
                    operands = "[%0], [%1, {{{}}}], [%2], %3".format(
                        ", ".join(["%{}".format(i + 4) for i in range(dim)])
                    )
                else:
                    operands = "[%0], [%1, {{{}}}], [%2]".format(", ".join(["%{}".format(i + 3) for i in range(dim)]))
                template = inst + " " + operands + ";"
                coords_type = meta.types([int32 for _ in range(dim)])
                cache_hint_type = meta.types([uint64] if has_cache_hint else [])

                @no_type_check
                @register_primitive_function_decorator
                @script
                def cp_async_tensor_global_to_shared_device(
                    dst: uint32,
                    tensor_map: CUTensorMapPointerType,
                    coords: coords_type,
                    mbarrier: uint32,
                    cache_hint: cache_hint_type,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = "cuda_internal"
                    asm(
                        template=template,
                        inputs=[dst, tensor_map, mbarrier, *cache_hint, *coords],
                        is_volatile=True,
                        memory_fence=True,
                    )

    for dim in [1, 2, 3, 4, 5]:
        for multicast in [False, True]:
            for cta_group in [None, 1, 2]:
                for has_cache_hint in [False, True]:
                    func_name = resolve_cp_async_bulk_tensor_global_to_cluster_shared(
                        dim=dim, multicast=multicast, cta_group=cta_group, cache_hint=has_cache_hint
                    )
                    multicast_item = "" if not multicast else ".multicast::cluster"
                    cta_group_item = "" if cta_group is None else ".cta_group::{}".format(cta_group)
                    cache_hint_item = "" if not has_cache_hint else "::L2::cache_hint"
                    inst = "cp.async.bulk.tensor.{}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes{}{}{}".format(
                        dim, multicast_item, cta_group_item, cache_hint_item
                    )
                    cnt = 0
                    operands = "[%{}]".format(cnt)  # dst
                    cnt += 1
                    operands += ", [%{}, {{{}}}]".format(
                        cnt, ", ".join(["%{}".format(i + cnt + 1) for i in range(dim)])
                    )  # tensor map and coords
                    cnt += 1 + dim
                    operands += ", [%{}]".format(cnt)  # mbarrier
                    cnt += 1
                    if multicast:
                        operands += ", %{}".format(cnt)  # multicast group
                        cnt += 1
                    if has_cache_hint:
                        operands += ", %{}".format(cnt)  # cache hint
                        cnt += 1
                    template = inst + " " + operands + ";"
                    coords_type = meta.types([int32 for _ in range(dim)])
                    cta_mask_type = meta.types([uint16] if multicast else [])
                    cache_hint_type = meta.types([uint64] if has_cache_hint else [])

                    @no_type_check
                    @register_primitive_function_decorator
                    @script
                    def cp_async_tensor_global_to_cluster_shared_device(
                        dst: uint32,
                        tensor_map: CUTensorMapPointerType,
                        coords: coords_type,
                        mbarrier: uint32,
                        cta_mask: cta_mask_type,
                        cache_hint: cache_hint_type,
                    ):
                        attrs.func_name = func_name
                        attrs.func_kind = "cuda_internal"
                        asm(
                            template=template,
                            inputs=[dst, tensor_map, *coords, mbarrier, *cta_mask, *cache_hint],
                            is_volatile=True,
                            memory_fence=True,
                        )

    for dim in [1, 2, 3, 4, 5]:
        for has_cache_hint in [False, True]:
            func_name = resolve_cp_async_bulk_tensor_shared_to_global(dim=dim, cache_hint=has_cache_hint)
            cache_hint_item = "" if not has_cache_hint else "::L2::cache_hint"
            inst = "cp.async.bulk.tensor.{}d.global.shared::cta.tile.bulk_group{}".format(dim, cache_hint_item)
            if has_cache_hint:
                operands = "[%0, {{{}}}], [%1], %2".format(", ".join(["%{}".format(i + 3) for i in range(dim)]))
            else:
                operands = "[%0, {{{}}}], [%1]".format(", ".join(["%{}".format(i + 2) for i in range(dim)]))
            template = inst + " " + operands + ";"
            coords_type = meta.types([int32 for _ in range(dim)])
            cache_hint_type = meta.types([uint64] if has_cache_hint else [])

            @no_type_check
            @register_primitive_function_decorator
            @script
            def cp_async_tensor_shared_to_global_device(
                dst_tensor_map: CUTensorMapPointerType, src: uint32, coords: coords_type, cache_hint: cache_hint_type
            ):
                attrs.func_name = func_name
                attrs.func_kind = "cuda_internal"
                asm(
                    template=template,
                    inputs=[dst_tensor_map, src, *cache_hint, *coords],
                    is_volatile=True,
                    memory_fence=True,
                )

    @no_type_check
    @register_primitive_function_decorator
    @script
    def cp_async_tensor_commit_group():
        attrs.func_name = "cp_async_tensor_commit_group"
        attrs.func_kind = "cuda_internal"
        asm(template="cp.async.bulk.commit_group;", inputs=[], is_volatile=True, memory_fence=True)

    for n in [0, 1, 2, 3, 4, 5, 6]:

        @no_type_check
        @register_primitive_function_decorator
        @script
        def func():
            attrs.func_name = "cp_async_tensor_wait_group_{}".format(n)
            attrs.func_kind = "cuda_internal"
            asm(template="cp.async.bulk.wait_group {};".format(n), inputs=[], is_volatile=True, memory_fence=True)


def cp_async_tensor_global_to_shared(
    dst: Expr,
    src_tensor_map: Expr,
    coords: Sequence[Expr],
    mbarrier: Expr,
    cta_group: Optional[int] = None,
    cache_policy: Optional[Expr] = None,
) -> Expr:
    """Perform a bulk copy from global memory to shared memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in shared memory. It should be an address with shared memory space with type uint32.
    src_tensor_map: Expr
        The tensor map that describes how the data is laid out in the global memory. It should be a pointer to
        an object with CUtensorMapType type.
    coords: Sequence[Expr]
        The coordinates of the tile to be copied.
    mbarrier: Expr
        The mbarrier to be used for synchronization. It should be an address with shared memory space with type uint32
        that has been initialized by `mbarrier_init`.
    cta_group: int, optional
        See the documentation of PTX of tma tensor copy.
    cache_policy: Expr, optional
        The cache policy to be used.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    assert cache_policy is None, "cache_policy is not supported yet"
    func_name = resolve_cp_async_bulk_tensor_global_to_shared(
        dim=len(coords), cta_group=cta_group, cache_hint=cache_policy is not None
    )
    optional_cache_policy: list[Expr] = [cache_policy] if cache_policy is not None else []
    return call_primitive_func(func_name, args=[dst, src_tensor_map, *coords, mbarrier, *optional_cache_policy])


def cp_async_tensor_global_to_cluster_shared(
    dst: Expr,
    src_tensor_map: Expr,
    coords: Sequence[Expr],
    mbarrier: Expr,
    multicast_mask: Expr,
    cta_group: Optional[int] = None,
    cache_policy: Optional[Expr] = None,
) -> Expr:
    """Perform a bulk copy from global memory to cluster shared memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in shared memory. It should be an address with shared memory space with type uint32.
    src_tensor_map: Expr
        The tensor map that describes how the data is laid out in the global memory. It should be a pointer to
        an object with CUtensorMapType type.
    coords: Sequence[Expr]
        The coordinates of the tile to be copied.
    mbarrier: Expr
        The mbarrier to be used for synchronization. It should be an address with shared memory space with type uint32
        that has been initialized by `mbarrier_init`.
    multicast_mask: Expr
        The multicast mask to be used. It should be a uint16 variable specifying the multicast group.
    cta_group: int, optional
        See the documentation of PTX of tma tensor copy.
    cache_policy: Expr, optional
        The cache policy to be used.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    assert cache_policy is None, "cache_policy is not supported yet"
    func_name = resolve_cp_async_bulk_tensor_global_to_cluster_shared(
        dim=len(coords),
        multicast=True,
        cta_group=cta_group,
        cache_hint=cache_policy is not None,
    )
    optional_cache_policy: list[Expr] = [cache_policy] if cache_policy is not None else []
    return call_primitive_func(
        func_name,
        args=[dst, src_tensor_map, *coords, mbarrier, multicast_mask, *optional_cache_policy],
    )


def cp_async_tensor_shared_to_global(
    dst_tensor_map: Expr, src: Expr, coords: Sequence[Expr], cache_policy: Optional[Expr] = None
) -> Expr:
    """Perform a bulk copy from shared memory to global memory asynchronously.

    Parameters
    ----------
    dst_tensor_map: Expr
        The tensor map that describes how the data is laid out in the global memory. It should be a pointer to
        an object with CUtensorMapType type.
    src: Expr
        The source address in shared memory. It should be an address with shared memory space with type uint32.
    coords: Sequence[Expr]
        The coordinates of the tile to be copied.
    cache_policy: Expr, optional
        The cache policy to be used.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    assert cache_policy is None, "cache_policy is not supported yet"
    func_name = resolve_cp_async_bulk_tensor_shared_to_global(dim=len(coords), cache_hint=cache_policy is not None)
    optional_cache_policy: list[Expr] = [cache_policy] if cache_policy is not None else []
    return call_primitive_func(func_name, args=[dst_tensor_map, src, *coords, *optional_cache_policy])


def cp_async_tensor_commit_group() -> Expr:
    """Commit the previously issued async tensor copy operations.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    return call_primitive_func("cp_async_tensor_commit_group", args=[])


def cp_async_tensor_wait_group(n: int) -> Expr:
    """Wait for the previously issued async tensor copy operations to complete.

    Parameters
    ----------
    n: int
        The number of groups to wait for. It should be an integer between 1 and 6.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    assert 0 <= n <= 6
    func_name = "cp_async_tensor_wait_group_{}".format(n)
    return call_primitive_func(func_name, args=[])
