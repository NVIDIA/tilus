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
"""Philox-4x32-10 PRNG primitive function."""

from typing import no_type_check

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.primitives.cuda.integer_intrinsics import umulhi
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.utils import initialize


@initialize()
def register_functions():
    from tilus.hidet.lang import attrs, script  # pylint: disable=import-outside-toplevel
    from tilus.hidet.lang.types import uint32

    p_u32 = ~uint32  # PointerType(uint32)

    # Philox-4x32 constants from Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (2011).
    # KEY_A/B are Weyl sequence constants (golden ratio and sqrt(3) based).
    # ROUND_A/B are multiplier constants selected to pass BigCrush in minimal rounds.
    PHILOX_KEY_A = 0x9E3779B9
    PHILOX_KEY_B = 0xBB67AE85
    PHILOX_ROUND_A = 0xD2511F53
    PHILOX_ROUND_B = 0xCD9E8D57

    @no_type_check
    @script
    def cuda_philox4x32(
        seed_lo: uint32,
        seed_hi: uint32,
        offset: uint32,
        out0: p_u32,
        out1: p_u32,
        out2: p_u32,
        out3: p_u32,
    ):
        """Philox-4x32-10 PRNG: given seed (split into lo/hi) and offset, write 4 random uint32 outputs."""
        attrs.func_kind = "cuda_internal"

        c0: uint32 = offset
        c1: uint32 = uint32(0)
        c2: uint32 = uint32(0)
        c3: uint32 = uint32(0)

        k0: uint32 = seed_lo
        k1: uint32 = seed_hi

        round_a: uint32 = uint32(PHILOX_ROUND_A)
        round_b: uint32 = uint32(PHILOX_ROUND_B)

        # 10 rounds (unrolled at Python level during script compilation)
        for _round in range(10):
            old_c0: uint32 = c0
            old_c2: uint32 = c2

            hi_b_c2: uint32 = umulhi(round_b, old_c2)
            hi_a_c0: uint32 = umulhi(round_a, old_c0)

            c0 = hi_b_c2 ^ c1 ^ k0
            c2 = hi_a_c0 ^ c3 ^ k1
            c1 = round_b * old_c2
            c3 = round_a * old_c0

            k0 = k0 + uint32(PHILOX_KEY_A)
            k1 = k1 + uint32(PHILOX_KEY_B)

        out0[0] = c0
        out1[0] = c1
        out2[0] = c2
        out3[0] = c3

    assert isinstance(cuda_philox4x32, Function)
    register_primitive_function(name=cuda_philox4x32.name, func_or_type=cuda_philox4x32)


def philox4x32(seed_lo: Expr, seed_hi: Expr, offset: Expr, out0: Expr, out1: Expr, out2: Expr, out3: Expr) -> Expr:
    """
    Philox-4x32-10 PRNG. Given a seed (split into lo/hi uint32 halves) and an offset (uint32),
    writes 4 random uint32 values to the output pointers.

    Parameters
    ----------
    seed_lo: Expr
        Lower 32 bits of the uint64 seed.
    seed_hi: Expr
        Upper 32 bits of the uint64 seed.
    offset: Expr
        The uint32 offset (counter value).
    out0, out1, out2, out3: Expr
        Pointers to uint32 where the 4 random outputs will be written.
    """
    return call_primitive_func("cuda_philox4x32", args=[seed_lo, seed_hi, offset, out0, out1, out2, out3])
