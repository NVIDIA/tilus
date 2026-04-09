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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

from tilus.hidet.ir.expr import Call, Expr
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.primitives.func import call_primitive_func, is_primitive_function, register_primitive_function
from tilus.hidet.ir.stmt import asm
from tilus.hidet.ir.type import FuncType, VoidType
from tilus.hidet.lang import attrs, script
from tilus.hidet.utils import initialize


@initialize()
def register_primitive_functions():
    functions = [
        ("cuda_syncthreads", "__syncthreads", FuncType([], VoidType())),
        ("cuda_syncthreads_count", "__syncthreads_count", FuncType(["int32"], "int32")),
        ("cuda_syncthreads_and", "__syncthreads_and", FuncType(["int32"], "int32")),
        ("cuda_syncthreads_or", "__syncthreads_or", FuncType(["int32"], "int32")),
        ("cuda_syncwarp", "__syncwarp", FuncType([], VoidType())),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def syncthreads() -> Call:
    return call_primitive_func("cuda_syncthreads", [])


def syncthreads_count(value: Expr) -> Call:
    return call_primitive_func("cuda_syncthreads_count", [value])


def syncthreads_and(cond: Union[Expr, int, bool]) -> Call:
    return call_primitive_func("cuda_syncthreads_and", [cond])


def syncthreads_or(cond: Expr) -> Call:
    return call_primitive_func("cuda_syncthreads_or", [cond])


def syncwarp() -> Call:
    return call_primitive_func("cuda_syncwarp", [])


def bar_sync(cooperative_threads: int) -> Call:
    """Synchronize threads using bar.sync with barrier 1.

    Deprecated: prefer bar_sync_aligned(barrier_id, thread_count) for explicit barrier ID control.
    """
    return bar_sync_aligned(barrier_id=1, thread_count=cooperative_threads)


def bar_sync_aligned(barrier_id: int, thread_count: int) -> Call:
    """Synchronize threads using a named barrier with bar.cta.sync.

    PTX: bar.cta.sync {barrier_id}, {thread_count};

    The .aligned modifier is used (via bar.cta.sync shorthand) to indicate all participating
    threads execute the same barrier instruction.

    Parameters
    ----------
    barrier_id : int
        Named barrier index (0-15). Barrier 0 is reserved for __syncthreads().
    thread_count : int
        Number of participating threads. Must be a multiple of 32.
    """
    if not 0 <= barrier_id <= 15:
        raise ValueError(f"barrier_id must be 0..15, got {barrier_id}")
    if thread_count % 32 != 0:
        raise ValueError(f"thread_count must be a multiple of 32, got {thread_count}")

    func_name = f"cuda_bar_sync_aligned_{barrier_id}_{thread_count}"
    if not is_primitive_function(func_name):

        @script
        def cuda_bar_sync_aligned():
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            template = "bar.cta.sync {}, {};".format(barrier_id, thread_count)
            asm(template=template)

        assert isinstance(cuda_bar_sync_aligned, Function)
        register_primitive_function(name=cuda_bar_sync_aligned.name, func_or_type=cuda_bar_sync_aligned)
    return call_primitive_func(func_name, [])


def bar_warp_sync(membermask: int) -> Call:
    """Synchronize a subset of threads within a warp.

    PTX: bar.warp.sync {membermask};

    Parameters
    ----------
    membermask : int
        32-bit mask where each bit corresponds to a lane in the warp.
        The executing thread must be in the mask.
    """
    func_name = f"cuda_bar_warp_sync_{membermask:#010x}"
    if not is_primitive_function(func_name):

        @script
        def cuda_bar_warp_sync():
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            template = "bar.warp.sync {};".format(membermask)
            asm(template=template)

        assert isinstance(cuda_bar_warp_sync, Function)
        register_primitive_function(name=cuda_bar_warp_sync.name, func_or_type=cuda_bar_warp_sync)
    return call_primitive_func(func_name, [])
