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
from hidet.ir.expr import Var, Expr, LogicalAnd, Equal, Mod, Constant
from tilus.ir.func import Function, Metadata
from tilus.ir.functors import IRRewriter
from tilus.ir.instructions import AssumeInst
from tilus.ir.layout import RegisterLayout, SharedLayout
from tilus.ir.layout.inference import infer_layout, verify_layouts
from tilus.ir.layout.inference.inference import LayoutInferenceError
from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.ir.tools import IRPrinter, rewrite
from tilus.transforms.base import Pass
from tilus.utils import gcd


class ApplyAssumeRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.params: tuple[Var, ...] = ()
        self.param2divisibility: dict[Var, int] = {}

    def visit_AssumeInst(self, inst: AssumeInst) -> None:
        # decompose the condition into conjuncture terms
        stack = [inst.condition]
        terms: list[Expr] = []
        while stack:
            expr = stack.pop()
            if isinstance(expr, LogicalAnd):
                stack.append(expr.a)
                stack.append(expr.b)
            else:
                terms.append(expr)

        # analyze the conjunctures
        for term in terms:
            # a % c == 0
            if (
                isinstance(term, Equal)
                and isinstance(term.a, Mod)
                and isinstance(term.a.b, Constant)
                and isinstance(term.a.a, Var)
                and isinstance(term.b, Constant)
                and term.b.value == 0
            ):
                a = term.a.a
                if a not in self.params:
                    raise RuntimeError(
                        "We only allow to specify the divisibility of kernel parameter, got {}".format(a.name)
                    )
                self.param2divisibility[a] = int(term.a.b.value)  # type: ignore[arg-type]
            else:
                raise RuntimeError("Can not recognize the condition in assume: {}".format(term))

    def visit_Function(self, func: Function) -> Function:
        self.params = func.params
        self.param2divisibility = {}

        updated_func = super().visit_Function(func)

        if updated_func is func and not self.param2divisibility:
            return func
        else:
            param2divisibility = updated_func.metadata.param2divisibility.copy()
            for var in self.param2divisibility:
                if var in param2divisibility:
                    param2divisibility[var] = gcd(param2divisibility[var], self.param2divisibility[var])
                else:
                    param2divisibility[var] = self.param2divisibility[var]
            return updated_func.with_metadata(
                updated_func.metadata.with_param2divisibility(param2divisibility)
            )


class LowerAssumePass(Pass):
    def process_function(self, func: Function) -> Function:
        apply_assume = ApplyAssumeRewriter()
        func = apply_assume(func)
        return func

def lower_assume_pass() -> Pass:
    return LowerAssumePass()

