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

from typing import Dict, List, Optional, Union

from hidet.ir.expr import Expr, Var
from hidet.ir.primitives.cuda.vars import blockIdx
from hidet.ir.type import BaseType

from tilus.ir.builders.stmt_builder import StmtBuilder
from tilus.ir.func import Function, Metadata
from tilus.ir.stmt import SeqStmt, Stmt
from tilus.ir.utils.normalize import normalize_cluster_blocks, normalize_grid_blocks


class FunctionBuilder(StmtBuilder):
    class _FunctionContext:
        def __init__(
            self, builder: FunctionBuilder, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]
        ) -> None:
            self.builder: FunctionBuilder = builder
            self.name: str = name
            self.num_warps: int = num_warps
            self.params: List[Var] = (
                [Var(name, type) for name, type in params.items()] if isinstance(params, dict) else params
            )

            self.builder.num_blocks = None
            self.builder.cluster_blocks = None
            self.builder._stack = [[]]

        def __enter__(self):
            if len(self.params) == 1:
                return self.params[0]
            else:
                return self.params

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                return

            assert len(self.builder._stack) == 1, len(self.builder._stack)

            if self.builder.num_blocks is None:
                raise ValueError("Please use `builder.num_blocks = ...` to set the number of blocks.")

            num_blocks: tuple[Expr, Expr, Expr] = normalize_grid_blocks(self.builder.num_blocks)
            if self.builder.cluster_blocks is not None:
                cluster_blocks: tuple[int, int, int] = normalize_cluster_blocks(self.builder.cluster_blocks)
            else:
                cluster_blocks = (1, 1, 1)

            built_function = Function.create(
                name=self.name,
                params=self.params,
                body=SeqStmt.create(self.builder._stack.pop()),
                metadata=Metadata.create(
                    grid_blocks=num_blocks,
                    cluster_blocks=cluster_blocks,
                    block_indices=[blockIdx.x, blockIdx.y, blockIdx.z],
                    num_warps=self.num_warps,
                ),
            )
            self.builder._on_finish(built_function)

    def __init__(self) -> None:
        super().__init__()
        self.num_blocks: Optional[List[Expr | int] | int | Expr] = None
        self.cluster_blocks: Optional[List[Expr | int] | int | Expr] = None

        # built function
        self._built_function: Optional[Function] = None

    def _reset(self):
        self._stack: list[list[Stmt]] = [[]]  # for StatementBuilder

        self._name = None
        self._params = []
        self._num_blocks = []

    def _on_finish(self, built_function: Function) -> None:
        self._built_function = built_function
        self._reset()

    def function(self, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]) -> _FunctionContext:
        return self._FunctionContext(self, name, num_warps, params)

    def flush_function(self) -> Function:
        assert self._built_function is not None
        ret = self._built_function
        self._built_function = None
        return ret
