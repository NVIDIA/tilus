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
"""
FastDivmod primitives for replacing integer division with multiply-high + shift.

Two primitive functions are registered:

1. ``fastdiv(a, b)`` — semantic placeholder for ``floor(a / b)`` where a >= 0 and b > 0.
   A Hidet IR pass (LowerFastDivPass) replaces this with the runtime computation using
   precomputed magic multiplier and shift.

2. ``fastdiv_runtime(a, multiplier, shift)`` — the actual runtime computation:
   ``__umulhi(a, multiplier) >> shift``. Emitted on the device side.

Host-side precomputation of (multiplier, shift) is done using plain C functions
(``fastdiv_precompute_m`` and ``fastdiv_precompute_s``) that run in the launch function.
"""

from typing import no_type_check

from tilus.hidet.ir.dtypes import uint32, uint64
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.ir.stmt import asm
from tilus.hidet.lang import attrs, script
from tilus.hidet.utils import initialize


@initialize()
def register_fast_divmod_primitives():
    @no_type_check
    @script
    def cuda_fastdiv(a: uint32, b: uint32) -> uint32:
        """Semantic placeholder for floor(a / b) where a >= 0 and b > 0.

        This function will be lowered by LowerFastDivPass to use precomputed
        magic multiplier and shift. It should never survive to codegen.
        """
        attrs.func_kind = "cuda_internal"
        return a / b  # fallback (should be lowered before codegen)

    @no_type_check
    @script
    def cuda_fastdiv_runtime(a: uint32, multiplier: uint32, shift: uint32) -> uint32:
        """Runtime fast division: __umulhi(a, multiplier) >> shift.

        When multiplier == 0, the divisor is a power of 2 and we use a simple shift.
        This runs on the device.
        """
        attrs.func_kind = "cuda_internal"
        ret: uint32 = 0
        if multiplier == uint32(0):
            ret = a >> shift
        else:
            asm(
                template="mul.hi.u32 %0, %1, %2; shr.u32 %0, %0, %3;",
                outputs=[ret],
                inputs=[a, multiplier, shift],
            )
        return ret

    @no_type_check
    @script
    def cuda_fastdiv_precompute_m(b: uint32) -> uint32:
        """Host-side: compute magic multiplier for divisor b.

        Uses the Hacker's Delight algorithm for unsigned division by constant.
        For power-of-2 divisors, returns 0 (sentinel) and the runtime uses a simple shift.
        For non-power-of-2 divisors, finds the smallest p >= 32 such that
        2^p > nc * (d - 2^p % d), then m = (2^p + d - 2^p % d) / d.
        """
        attrs.func_kind = "cpu_internal"
        # Check power of 2: b & (b - 1) == 0
        if (b & (b - uint32(1))) == uint32(0):
            return uint32(0)
        nc: uint64 = (uint64(1) << uint64(32)) - uint64(1) - uint64((uint64(1) << uint64(32)) % uint64(b))
        p: uint32 = 32
        while (uint64(1) << uint64(p)) <= nc * uint64(uint64(b) - (uint64(1) << uint64(p)) % uint64(b)):
            p = p + uint32(1)
        m: uint64 = ((uint64(1) << uint64(p)) + uint64(b) - (uint64(1) << uint64(p)) % uint64(b)) / uint64(b)
        return uint32(m)

    @no_type_check
    @script
    def cuda_fastdiv_precompute_s(b: uint32) -> uint32:
        """Host-side: compute shift for divisor b.

        For power-of-2 divisors, returns log2(b).
        For non-power-of-2 divisors, returns p - 32 where p is from the Hacker's Delight algorithm.
        """
        attrs.func_kind = "cpu_internal"
        if (b & (b - uint32(1))) == uint32(0):
            # Power of 2: return log2(b)
            s: uint32 = 0
            tmp: uint32 = b >> uint32(1)
            while tmp > uint32(0):
                tmp = tmp >> uint32(1)
                s = s + uint32(1)
            return s
        nc: uint64 = (uint64(1) << uint64(32)) - uint64(1) - uint64((uint64(1) << uint64(32)) % uint64(b))
        p: uint32 = 32
        while (uint64(1) << uint64(p)) <= nc * uint64(uint64(b) - (uint64(1) << uint64(p)) % uint64(b)):
            p = p + uint32(1)
        return p - uint32(32)

    for func in [cuda_fastdiv, cuda_fastdiv_runtime, cuda_fastdiv_precompute_m, cuda_fastdiv_precompute_s]:
        register_primitive_function(name=func.name, func_or_type=func)


def fastdiv(a: Expr, b: Expr) -> Expr:
    """Semantic floor division: floor(a / b) for a >= 0, b > 0.

    Will be lowered by LowerFastDivPass to use precomputed multiplier and shift.
    """
    return call_primitive_func("cuda_fastdiv", args=[a, b])


def fastdiv_runtime(a: Expr, multiplier: Expr, shift: Expr) -> Expr:
    """Runtime fast division using precomputed multiplier and shift (device-side)."""
    return call_primitive_func("cuda_fastdiv_runtime", args=[a, multiplier, shift])


def fastdiv_precompute_m(b: Expr) -> Expr:
    """Precompute magic multiplier for divisor b (host-side)."""
    return call_primitive_func("cuda_fastdiv_precompute_m", args=[b])


def fastdiv_precompute_s(b: Expr) -> Expr:
    """Precompute shift amount for divisor b (host-side)."""
    return call_primitive_func("cuda_fastdiv_precompute_s", args=[b])
