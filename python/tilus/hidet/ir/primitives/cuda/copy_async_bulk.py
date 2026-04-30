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
from typing import Optional, no_type_check

from tilus.hidet.ir.dtypes import int32, uint16, uint32
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.primitives import register_primitive_function
from tilus.hidet.ir.primitives.cuda.funcs import call_cuda
from tilus.hidet.ir.type import void_p
from tilus.hidet.utils import initialize


def resolve_cp_async_bulk_global_to_shared(l2_evict: Optional[str]) -> str:
    if l2_evict not in [None, "evict_first"]:
        raise ValueError("l2_evict should be None or 'evict_first'")
    func_name = "cp_async_bulk_shared_global{}".format("_l2_evict_first" if l2_evict == "evict_first" else "")
    return func_name


def resolve_cp_async_bulk_global_to_cluster_shared(l2_evict: Optional[str]) -> str:
    if l2_evict not in [None, "evict_first"]:
        raise ValueError("l2_evict should be None or 'evict_first'")
    func_name = "cp_async_bulk_cluster_shared_global{}".format("_l2_evict_first" if l2_evict == "evict_first" else "")
    return func_name


def resolve_cp_async_bulk_shared_to_global(l2_evict: Optional[str], cp_mask: bool) -> str:
    if l2_evict not in [None, "evict_first"]:
        raise ValueError("l2_evict should be None or 'evict_first'")
    func_name = "cp_async_bulk_global_shared{}".format("_l2_evict_first" if l2_evict == "evict_first" else "")
    if cp_mask:
        func_name += "_cp_mask"
    return func_name


@initialize()
def register_bulk_copy_async():
    from tilus.hidet.lang import asm, attrs, script

    # cp_async_bulk_global_to_shared
    for l2_evict in [None, "evict_first"]:
        func_name = resolve_cp_async_bulk_global_to_shared(l2_evict)
        if l2_evict == "evict_first":
            template_string = (
                "{"
                "    .reg .b64 p;"
                "    createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
                "    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], p;"
                "}"
            )
        else:
            template_string = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        func_name = "cuda_" + func_name

        @no_type_check
        @script
        def cuda_cp_async(dst: uint32, src: void_p, size: int32, mbarrier: uint32):
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            asm(template=template_string, inputs=[dst, src, size, mbarrier], is_volatile=True, memory_fence=True)

        assert isinstance(cuda_cp_async, Function)
        register_primitive_function(name=func_name, func_or_type=cuda_cp_async)

    # cp_async_bulk_global_to_cluster_shared
    for l2_evict in [None, "evict_first"]:
        func_name = resolve_cp_async_bulk_global_to_cluster_shared(l2_evict)
        if l2_evict == "evict_first":
            template_string = (
                "{"
                "    .reg .b64 p;"
                "    createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
                "    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1], %2, [%3], %4, p;"
                "}"
            )
        else:
            template_string = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
        func_name = "cuda_" + func_name

        @no_type_check
        @script
        def cuda_cp_async(dst: uint32, src: void_p, size: int32, mbarrier: uint32, cta_mask: uint16):
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            asm(
                template=template_string,
                inputs=[dst, src, size, mbarrier, cta_mask],
                is_volatile=True,
                memory_fence=True,
            )

        assert isinstance(cuda_cp_async, Function)
        register_primitive_function(name=func_name, func_or_type=cuda_cp_async)

    # cp_async_bulk_s2g
    for l2_evict in [None, "evict_first"]:
        for cp_mask in [False, True]:
            func_name = resolve_cp_async_bulk_shared_to_global(l2_evict, cp_mask)

            operand_count = 3
            inst = "cp.async.bulk.global.shared::cta.bulk_group{cache_hint}{cp_mask} [%0], [%1], %2".format(
                cache_hint=".L2::cache_hint" if l2_evict is not None else "",
                cp_mask=".cp_mask" if cp_mask else "",
            )
            if l2_evict is not None:
                inst = inst + ", p"
            if cp_mask:
                inst = inst + ", %{}".format(operand_count)
                operand_count += 1
            if l2_evict == "evict_first":
                template_string = (
                    "{    .reg .b64 p;    createpolicy.fractional.L2::evict_first.b64 p, 1.0;    " + inst + ";}"
                )
            else:
                template_string = inst + ";"

            func_name = "cuda_" + func_name
            if not cp_mask:

                @no_type_check
                @script
                def cuda_cp_async(dst: void_p, src: uint32, size: int32):
                    attrs.func_name = func_name
                    attrs.func_kind = "cuda_internal"
                    inputs = [dst, src, size]
                    asm(template=template_string, inputs=inputs, is_volatile=True, memory_fence=True)
            else:

                @no_type_check
                @script
                def cuda_cp_async(dst: void_p, src: uint32, size: int32, byte_mask: uint32):
                    attrs.func_name = func_name
                    attrs.func_kind = "cuda_internal"
                    inputs = [dst, src, size, byte_mask]
                    asm(template=template_string, inputs=inputs, is_volatile=True, memory_fence=True)

            assert isinstance(cuda_cp_async, Function)
            register_primitive_function(name=func_name, func_or_type=cuda_cp_async)

    # cp_async_shared_to_cluster_shared
    func_name = "cuda_cp_async_bulk_cluster_shared_shared"

    @no_type_check  # type: ignore[no-redef]
    @script
    def cuda_cp_async(dst: uint32, src: uint32, size: int32, mbarrier: uint32):
        attrs.func_name = func_name
        attrs.func_kind = "cuda_internal"
        asm(
            template="cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];",
            inputs=[dst, src, size, mbarrier],
            is_volatile=True,
            memory_fence=True,
        )

    assert isinstance(cuda_cp_async, Function)
    register_primitive_function(name=func_name, func_or_type=cuda_cp_async)


def cp_async_bulk_global_to_shared(
    dst: Expr, src: Expr, size: Expr, mbarrier: Expr, l2_evict: Optional[str] = None
) -> Expr:
    func_name = resolve_cp_async_bulk_global_to_shared(l2_evict)
    return call_cuda(func_name, args=[dst, src, size, mbarrier])


def cp_async_bulk_global_to_cluster_shared(
    dst: Expr, src: Expr, size: Expr, mbarrier: Expr, cta_mask: int, l2_evict: Optional[str] = None
) -> Expr:
    func_name = resolve_cp_async_bulk_global_to_cluster_shared(l2_evict)
    return call_cuda(func_name, args=[dst, src, size, mbarrier, uint32(cta_mask)])


def cp_async_bulk_shared_to_global(
    dst: Expr,
    src: Expr,
    size: Expr,
    l2_evict: Optional[str] = None,
    byte_mask: Optional[Expr] = None,
) -> Expr:
    func_name = resolve_cp_async_bulk_shared_to_global(l2_evict, byte_mask is not None)
    args = [dst, src, size]
    if byte_mask is not None:
        args.append(byte_mask)
    return call_cuda(func_name, args=args)


def cp_async_bulk_shared_to_cluster_shared(
    dst: Expr,
    src: Expr,
    size: Expr,
    mbarrier: Expr,
) -> Expr:
    func_name = "cp_async_bulk_cluster_shared_shared"
    return call_cuda(func_name, args=[dst, src, size, mbarrier])
