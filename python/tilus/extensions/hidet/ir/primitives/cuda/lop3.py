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
from typing import Callable, Union

from hidet.ir.dtypes import uint32
from hidet.ir.expr import Expr, cast
from hidet.ir.stmt import asm


def lop3(d: Expr, a: Expr, b: Expr, c: Expr, *, imm_lut: Union[int, Callable[[int, int, int], int]]) -> Expr:
    """
    Perform a logical operation on three 32-bit values and store the result in `d`.

    The logical operation is determined by the immediate value `imm_lut`.

    See the PTX ISA documentation for the `lop3` instruction for more information:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3

    Parameters
    ----------
    d: Expr
        The pointer to the 32-bit result.
    a: Expr
        The first 32-bit operand.
    b: Expr
        The second 32-bit operand.
    c: Expr
        The third 32-bit operand.
    imm_lut: int
        The immediate value that determines the logical operation. Given logical operation `f(a, b, c)`, the
        immediate value `imm_lut` should be set to `f(0xF0, 0xCC, 0xAA)` to indicate the logical operation.
    """
    if not isinstance(imm_lut, int):
        imm_lut = imm_lut(0xF0, 0xCC, 0xAA)

    assert 0 <= imm_lut <= 255

    return asm(
        "lop3.b32 %0, %1, %2, %3, {};".format(imm_lut),
        outputs=[cast(d, ~uint32)[0]],
        inputs=[a, b, c],
        is_volatile=True,
    )
