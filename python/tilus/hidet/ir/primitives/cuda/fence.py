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
from typing import no_type_check

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.ir.stmt import asm
from tilus.hidet.lang import attrs, script
from tilus.hidet.utils import initialize


@initialize()
def register_mbarrier_primitives():
    @no_type_check
    @script
    def cuda_fence_mbarrier_init_cluster():
        attrs.func_kind = "cuda_internal"
        asm("fence.mbarrier_init.release.cluster;", is_volatile=True, memory_fence=True)

    # fence.proxy.async.{space}
    for space, ptx_space in [("shared", "shared::cta"), ("global", "global")]:
        func_name = "cuda_fence_proxy_async_{}".format(space)
        inst = "fence.proxy.async.{};".format(ptx_space)

        @no_type_check
        @script
        def cuda_fence_proxy_async():
            attrs.func_kind = "cuda_internal"
            attrs.func_name = func_name
            asm(template=inst, inputs=[], is_volatile=True, memory_fence=True)

        register_primitive_function(name=func_name, func_or_type=cuda_fence_proxy_async)

    @no_type_check
    @script
    def cuda_fence_proxy_async_generic_release_shared():
        attrs.func_kind = "cuda_internal"
        asm(
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
            is_volatile=True,
            memory_fence=True,
        )

    for func in [cuda_fence_mbarrier_init_cluster, cuda_fence_proxy_async_generic_release_shared]:
        register_primitive_function(func.name, func)


def fence_mbarrier_init_cluster() -> Expr:
    """Issue a fence to initialize mbarrier in cluster scope.

    Returns
    -------
    Expr
        An expression representing the fence operation.
    """
    return call_primitive_func("cuda_fence_mbarrier_init_cluster", args=[])


def fence_proxy_async(space: str) -> Expr:
    """
    Emit a bidirectional proxy fence for async memory operations.

    PTX: fence.proxy.async.{space}

    Parameters
    ----------
    space : str
        The space of the fence: 'shared' for fence.proxy.async.shared::cta,
        'global' for fence.proxy.async.global.

    Returns
    -------
    ret : Expr
        A call expression that performs the fence operation.
    """
    func_name = "cuda_fence_proxy_async_{}".format(space)
    return call_primitive_func(func_name, args=[])


def fence_proxy_async_generic_release_shared() -> Expr:
    """
    Emit a unidirectional generic-to-async release proxy fence for shared memory.

    PTX: fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster

    This is a lighter-weight fence that only ensures prior generic proxy writes
    to shared::cta memory are visible to subsequent async proxy reads.

    Returns
    -------
    ret : Expr
        A call expression that performs the fence operation.
    """
    return call_primitive_func("cuda_fence_proxy_async_generic_release_shared", args=[])
