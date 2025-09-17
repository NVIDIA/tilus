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

from dataclasses import dataclass
from typing import Optional

from hidet.ir.expr import Expr, Var
from hidet.ir.primitives.cuda.cluster import this_cluster
from hidet.ir.primitives.cuda.vars import blockIdx, gridDim
from hidet.ir.tools import collect
from hidet.ir.type import DataType

from tilus.backends.codegen import BaseEmitContext, FunctionCodegen, register_emit_context
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir import GlobalLayout


@dataclass
class GlobalTensorView:
    ptr: Var
    dtype: DataType
    layout: GlobalLayout


@register_emit_context
class InvariantTrackingContext(BaseEmitContext):
    """Context used to track grid- and block-level invariants."""

    _current: Optional[InvariantTrackingContext] = None

    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)
        self.grid_invariants: set[Var] = set()
        self.block_invariants: set[Var] = set()
        self.var2expr: dict[Var, Expr] = {}

    @staticmethod
    def current() -> InvariantTrackingContext:
        if InvariantTrackingContext._current is None:
            raise RuntimeError("No InvariantTrackingContext is currently active.")
        return InvariantTrackingContext._current

    def initialize(self):
        grid_invariant_vars = []
        block_invariant_vars = []

        for param in self.codegen.function.params:
            grid_invariant_vars.append(param)

        grid_invariant_vars.extend(
            [
                gridDim.x,  # type: ignore
                gridDim.y,  # type: ignore
                gridDim.z,  # type: ignore
                this_cluster.dim_blocks,  # type: ignore
                this_cluster.dim_threads,  # type: ignore
            ]
        )
        block_invariant_vars.extend(
            [
                blockIdx.x,  # type: ignore
                blockIdx.y,  # type: ignore
                blockIdx.z,  # type: ignore
            ]
        )
        self.initialize_grid_invariant_vars(grid_invariant_vars)
        self.initialize_block_invariant_vars(block_invariant_vars)

    def initialize_grid_invariant_vars(self, grid_invariants: list[Var]) -> None:
        self.grid_invariants.update(grid_invariants)
        self.block_invariants.update(grid_invariants)  # grid invariants are also block invariants
        for var in grid_invariants:
            if var not in self.var2expr:
                self.var2expr[var] = var

    def initialize_block_invariant_vars(self, block_invariants: list[Var]) -> None:
        self.block_invariants.update(block_invariants)
        for var in block_invariants:
            if var not in self.var2expr:
                self.var2expr[var] = var

    def bind(self, var: Var, value: Expr) -> None:
        used_vars = collect(value, Var, stop_when_found=True)
        is_block_invariant = True
        is_grid_invariant = True
        for v in used_vars:
            if v not in self.block_invariants:
                is_block_invariant = False
            if v not in self.grid_invariants:
                is_grid_invariant = False
        if is_block_invariant:
            self.block_invariants.add(var)
        if is_grid_invariant:
            self.grid_invariants.add(var)
        if is_block_invariant or is_grid_invariant:
            self.var2expr[var] = rewrite(value, self.var2expr)

    def rewrite_to_grid_invariant(self, expr: Expr) -> Expr:
        used_vars = collect(expr, Var, stop_when_found=True)
        if all(v in self.grid_invariants for v in used_vars):
            return rewrite(expr, self.var2expr)
        else:
            raise ValueError("Expression is not grid-invariant:\n{}".format(expr))

    def rewrite_to_block_invariant(self, expr: Expr) -> Expr:
        used_vars = collect(expr, Var, stop_when_found=True)
        if all(v in self.block_invariants for v in used_vars):
            return rewrite(expr, self.var2expr)
        else:
            raise ValueError("Expression is not block-invariant:\n{}".format(expr))
