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
from __future__ import annotations

from typing import Generic, Type, TypeVar

from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import Stmt as HidetStmt

T = TypeVar("T")


class BaseEmitContext(Generic[T]):
    REGISTRY: list[Type[BaseEmitContext]] = []

    def __init__(self, codegen):
        from tilus.backends.codegen import FunctionCodegen

        self.codegen: FunctionCodegen = codegen

        self.__post_init__()

    def __post_init__(self):
        pass

    @property
    def contexts(self):
        return self.codegen.contexts

    def host_prepend(self, stmt: Expr | HidetStmt) -> None:
        """Prepend a statement to the host function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be prepended.
        """
        self.codegen.host_builder.scope_stack[0].insert(0, stmt)

    def host_append(self, stmt: Expr | HidetStmt) -> None:
        """Append a statement to the host function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be appended.
        """
        self.codegen.host_builder.append(stmt)

    def kernel_prepend(self, stmt: Expr | HidetStmt) -> None:
        """Prepend a statement to the kernel function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be prepended.
        """
        self.codegen.builder.scope_stack[0].insert(0, stmt)

    def kernel_append(self, stmt: Expr | HidetStmt) -> None:
        """Append a statement to the kernel function.

        Parameters
        ----------
        stmt: Expr or HidetStmt
            The statement to be appended.
        """
        self.codegen.builder.append(stmt)

    def append_extra_param(self, var: Var) -> None:
        """Append an extra parameter to the kernel function.

        This method marks a variable in the host function to be passed as an extra parameter to the kernel function.
        The `var` must be a variable defined in the host function. The kernel function can directly use the `var` in the
        kernel body after this method is called.
        """
        self.codegen.extra_params.append(var)

    def initialize(self):
        """Initialize the context.

        This method is called before the codegen starts for all instructions.
        """
        pass

    def finalize(self):
        """Finalize the context.

        This method is called when the codegen is finished for all instructions.
        """
        pass
