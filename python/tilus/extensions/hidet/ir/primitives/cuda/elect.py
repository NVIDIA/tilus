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
from typing import no_type_check

from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.stmt import asm
from hidet.lang import script
from hidet.utils import initialize


@initialize()
def register_warp_uniform_primitives():
    from hidet.lang import attrs, i32, u32

    # elect.sync — elect one thread per warp
    elect_func_name = "cuda_elect_sync"

    @no_type_check
    @script
    def elect_sync(membermask: u32) -> u32:
        """Returns 1 for exactly one elected thread in the warp, 0 for all others.

        Maps to PTX: elect.sync laneid|pred, membermask;
        """
        attrs.func_name = elect_func_name
        attrs.func_kind = "cuda_internal"
        ret: u32 = 0
        asm(
            template="{ .reg.pred %%p; elect.sync _|%%p, %1; selp.u32 %0, 1, 0, %%p; }",
            outputs=[ret],
            inputs=[membermask],
        )
        return ret

    register_primitive_function(name=elect_sync.name, func_or_type=elect_sync)

    # shfl.sync with fixed i32 types — avoids Hidet's type_infer_func which has param_types=None
    shfl_func_name = "cuda_shfl_sync_i32"

    @no_type_check
    @script
    def shfl_sync_i32(mask: u32, val: i32, src_lane: i32) -> i32:
        """Broadcast src_lane's value of val to all threads in the warp.

        Maps to PTX: shfl.sync.idx.b32
        Fixed-type version of hidet's shfl_sync for compatibility with Tilus passes.
        """
        attrs.func_name = shfl_func_name
        attrs.func_kind = "cuda_internal"
        ret: i32 = 0
        asm(
            template="shfl.sync.idx.b32 %0, %2, %3, 31, %1;",
            outputs=[ret],
            inputs=[mask, val, src_lane],
        )
        return ret

    register_primitive_function(name=shfl_sync_i32.name, func_or_type=shfl_sync_i32)


def elect_one_sync(membermask: Expr = None) -> Expr:  # type: ignore[assignment]
    """Elect one thread from the warp.

    Returns a boolean-like u32 expression: 1 for the elected thread, 0 for all others.
    Uses the PTX ``elect.sync`` instruction.

    Parameters
    ----------
    membermask : Expr, optional
        The thread mask. Default: 0xFFFFFFFF (all threads).

    Returns
    -------
    ret : Expr
        u32 value, 1 for the elected thread, 0 for others.
    """
    if membermask is None:
        from hidet.ir.dtypes import uint32
        from hidet.ir.expr import Constant

        membermask = Constant(0xFFFFFFFF, uint32)
    return call_cuda("elect_sync", args=[membermask])


def shfl_sync_i32(mask: Expr, val: Expr, src_lane: Expr) -> Expr:
    """Broadcast src_lane's value to all threads in the warp (fixed i32 types).

    Parameters
    ----------
    mask : Expr
        Active thread mask (u32).
    val : Expr
        Value to broadcast (i32).
    src_lane : Expr
        Source lane index (i32).

    Returns
    -------
    ret : Expr
        The broadcast value (i32).
    """
    return call_cuda("shfl_sync_i32", args=[mask, val, src_lane])
