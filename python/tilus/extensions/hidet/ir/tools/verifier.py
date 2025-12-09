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
# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from hidet.ir import DeclareStmt, ForMappingStmt, ForStmt, Function, LetStmt
from hidet.ir.expr import Constant, Expr, Var
from hidet.ir.functors import IRVisitor
from hidet.ir.module import IRModule
from hidet.ir.node import Node
from hidet.ir.tools.printer import IRPrinter
from hidet.ir.tools.simplifier import Simplifier
from hidet.ir.type import FuncType
from hidet.utils.doc import Doc, NewLine


@dataclass
class Diagnostic:
    kind: str
    context: Node
    message: str
    items: Sequence[Any]

    def __init__(self, kind: str, context: Node, message: str, items: Sequence[Any]):
        assert kind == "error"
        if len(items) != message.count("{}"):
            raise ValueError("Number of items does not match number of placeholders in message.")
        for item in items:
            assert (
                isinstance(item, Node)
                or isinstance(item, (int, float, bool, str, Expr))
                or isinstance(item, (list, tuple, dict))
            ), type(item)
        self.kind = kind
        self.context = context
        self.message = message
        self.items = items


class Diagnostics:
    def __init__(self, module: IRModule, diagnostics: Sequence[Diagnostic]):
        self.module: IRModule = module
        self.diagnostics: list[Diagnostic] = list(diagnostics)

    def __bool__(self):
        return bool(self.diagnostics)

    def __str__(self):
        printer = IRPrinter()
        printer(self.module)
        kind_count: dict[str, int] = defaultdict(int)

        doc = Doc()
        doc += "Verification failed with {} errors:".format(len(self.diagnostics))
        doc += NewLine() + printer(self.module)
        for diag in self.diagnostics:
            doc += NewLine() + "In " + printer(diag.context) + ":"
            components = diag.message.split("{}")
            doc += NewLine() + "{} {}: ".format(diag.kind.capitalize(), kind_count[diag.kind] + 1)
            kind_count[diag.kind] += 1

            items = list(diag.items)
            for idx, component in enumerate(components):
                doc += component
                if idx == len(components) - 1:
                    continue
                doc += printer(items.pop(0))
            doc += NewLine()

        return str(doc)


class VerificationError(Exception):
    def __init__(self, diagnostics: Diagnostics):
        self.diagnostics = diagnostics

    def __str__(self):
        return str(self.diagnostics)


class IRVerifier(IRVisitor):
    def __init__(self):
        super().__init__()
        self.simplifier: Simplifier = Simplifier()
        self.diagnostics: list[Diagnostic] = []

        self.builtin_names: set[str] = set()
        self.scopes: list[list[Var]] = []

    @contextlib.contextmanager
    def new_scope(self):
        self.scopes.append([])
        yield
        self.scopes.pop()

    @property
    def current_scope(self) -> list[Var]:
        if not self.scopes:
            raise RuntimeError("No current scope available.")
        return self.scopes[-1]

    def const_or_zero(self, expr: Expr | int) -> int:
        expr = self.simplifier(expr)
        if isinstance(expr, Constant):
            assert isinstance(expr.value, int)
            return int(expr.value)
        return 0

    def error(self, context: Node, message: str, *items: Any) -> None:
        self.append("error", context, message, *items)

    def append(self, kind: str, context: Node, message: str, *items: Any) -> None:
        self.diagnostics.append(Diagnostic(kind, context, message, items))

    def defined(self, var: Var) -> bool:
        """Check if a variable is defined in the current scope."""
        return any(var in scope for scope in self.scopes) or var.name in self.builtin_names

    def define(self, var: Var, ctx: Node) -> None:
        if self.defined(var):
            self.error(ctx, "Variable {} is already defined.", var)
        else:
            self.current_scope.append(var)

    def visit_Function(self, func: Function) -> None:
        # define global primitive variables
        with self.new_scope():
            if func.kind in ["cuda_kernel", "cuda_internal"]:
                from hidet.ir.primitives.cuda.vars import blockDim, blockIdx, gridDim, threadIdx
                from tilus.extensions.hidet.ir.primitives.cuda.vars import clusterBlockIdx, clusterBlockRank, clusterDim, clusterIdx, clusterSize

                for var in [
                    threadIdx.x,
                    threadIdx.y,
                    threadIdx.z,
                    blockIdx.x,
                    blockIdx.y,
                    blockIdx.z,
                    blockDim.x,
                    blockDim.y,
                    blockDim.z,
                    gridDim.x,
                    gridDim.y,
                    gridDim.z,
                    clusterBlockIdx.x,
                    clusterBlockIdx.y,
                    clusterBlockIdx.z,
                    clusterBlockRank,
                    clusterDim.x,
                    clusterDim.y,
                    clusterDim.z,
                    clusterIdx.x,
                    clusterIdx.y,
                    clusterIdx.z,
                    clusterSize,
                ]:
                    self.builtin_names.add(var.name)
            elif func.kind == "public":
                pass
            else:
                raise NotImplementedError(func.kind)

            # define function parameters
            for param in func.params:
                self.define(param, ctx=func)

            self.visit(func.body)

    def visit_Var(self, e: Var) -> None:
        if isinstance(e.type, FuncType):
            # we assume that functions have been defined
            return
        if not self.defined(e):
            self.error(e, "Variable {} is not defined.", e)
        super().visit_Var(e)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        with self.new_scope():
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                self.visit(bind_value)
                self.define(bind_var, ctx=stmt)
            self.visit(stmt.body)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.visit(stmt.init)
        self.define(stmt.var, ctx=stmt)

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        self.visit(stmt.extent)
        with self.new_scope():
            self.define(stmt.loop_var, stmt)
            self.visit(stmt.body)

    def visit_ForTaskStmt(self, stmt: ForMappingStmt) -> None:
        self.visit(stmt.mapping)
        with self.new_scope():
            for var in stmt.loop_vars:
                self.define(var, ctx=stmt)
            self.visit(stmt.body)


def verify(ir_module: IRModule) -> None:
    verifier = IRVerifier()
    verifier(ir_module)
    if verifier.diagnostics:
        raise VerificationError(Diagnostics(module=ir_module, diagnostics=verifier.diagnostics))
