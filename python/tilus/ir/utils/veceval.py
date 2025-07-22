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

import numpy as np
from hidet.ir.expr import Add, Constant, Div, Expr, Mod, Multiply, Sub, Var
from hidet.ir.functors import IRFunctor


class VectorizedEvaluator(IRFunctor):
    def __init__(self, var2value: dict[Var, np.ndarray]):
        super().__init__()
        self.var2value = var2value

    def visit_Var(self, e: Var) -> np.ndarray:
        if e not in self.var2value:
            raise ValueError("Variable not found in var2value dictionary: {}".format(e))
        return self.var2value[e]

    def visit_Constant(self, e: Constant) -> np.ndarray:
        return np.asarray(e.value)

    def visit_Add(self, e: Add) -> np.ndarray:
        return self.visit(e.a) + self.visit(e.b)

    def visit_Sub(self, e: Sub) -> np.ndarray:
        return self.visit(e.a) - self.visit(e.b)

    def visit_Multiply(self, e: Multiply) -> np.ndarray:
        return self.visit(e.a) * self.visit(e.b)

    def visit_Div(self, e: Div) -> np.ndarray:
        a = self.visit(e.a)
        b = self.visit(e.b)
        is_integer = all(np.issubdtype(operand.dtype, np.integer) for operand in [a, b])
        if is_integer:
            return np.floor_divide(self.visit(e.a), self.visit(e.b))
        else:
            return np.true_divide(self.visit(e.a), self.visit(e.b))

    def visit_Mod(self, e: Mod) -> np.ndarray:
        return np.mod(self.visit(e.a), self.visit(e.b))


def vectorized_evaluate(expr: Expr, var2value: dict[Var, np.ndarray]) -> np.ndarray:
    evaluator = VectorizedEvaluator(var2value)
    return evaluator.visit(expr)


def meshgrid(shape: Sequence[int]) -> list[np.ndarray]:
    """
    Create a meshgrid for the given shape.
    """
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return [g.astype(np.int32) for g in grid]  # Convert to int32 for consistency with hidet types


def demo_meshgrid():
    shape = (3, 4)
    grid = meshgrid(shape)
    for i, g in enumerate(grid):
        print(f"Grid {i}:")
        print(g)
        print()


if __name__ == "__main__":
    demo_meshgrid()
