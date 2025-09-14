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
from typing import Sequence

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.ir.tools.simplifier import simplify, simplify_to_int


def normalize_dim3(dim3: Expr | int | Sequence[Expr | int]) -> tuple[Expr, Expr, Expr]:
    return normalize_grid_blocks(dim3)


def normalize_grid_blocks(blocks: Expr | int | Sequence[Expr | int]) -> tuple[Expr, Expr, Expr]:
    """Normalize grid blocks to a tuple of three expressions.

    Parameters
    ----------
    blocks: Expr | int | Sequence[Expr | int]
        The number of blocks in the grid. It can be an integer, an expression, or a list/tuple of integers or
        expressions. If it is an integer or expression, it represents the number of blocks in the x dimension, and the
        y and z dimensions are set to 1. If it is a list or tuple, it should have a length of 1, 2, or 3. If the length
        is less than 3, the missing dimensions are set to 1.

    Returns
    -------
    ret: tuple[Expr, Expr, Expr]
        The normalized number of blocks in the x, y, and z dimensions.
    """
    if isinstance(blocks, (Expr, int)):
        blocks_x: Expr = simplify(blocks)
        return blocks_x, int32.one, int32.one
    elif isinstance(blocks, (list, tuple)):
        blocks_expr = [simplify(b) for b in blocks]
        while len(blocks_expr) < 3:
            blocks_expr.append(int32.one)
        if len(blocks_expr) > 3:
            raise ValueError("The length of blocks_per_cluster should not be greater than 3.")
        return blocks_expr[0], blocks_expr[1], blocks_expr[2]
    else:
        raise ValueError("The type of blocks_per_cluster should be int, Expr, list or tuple.")


def normalize_cluster_blocks(blocks: Expr | int | Sequence[Expr | int]) -> tuple[int, int, int]:
    """Normalize cluster blocks to a tuple of three integers.

    Parameters
    ----------
    blocks: int or Expr or list or tuple
        The number of blocks per cluster. It can be an integer, an expression, or a list/tuple of integers or
        expressions. If it is an integer or expression, it represents the number of blocks in the x dimension, and the
        y and z dimensions are set to 1. If it is a list or tuple, it should have a length of 1, 2, or 3. If the length
        is less than 3, the missing dimensions are set to 1.

    Returns
    -------
    ret: tuple[int, int, int]
        The normalized number of blocks per cluster in the x, y, and z dimensions.
    """
    if isinstance(blocks, (Expr, int)):
        return simplify_to_int(blocks), 1, 1
    elif isinstance(blocks, (list, tuple)):
        blocks_int = [simplify_to_int(b) for b in blocks]
        while len(blocks_int) < 3:
            blocks_int.append(1)
        if len(blocks_int) > 3:
            raise ValueError("The length of blocks_per_cluster should not be greater than 3.")
        return blocks_int[0], blocks_int[1], blocks_int[2]
    else:
        raise ValueError("The type of blocks_per_cluster should be int, Expr, list or tuple.")
