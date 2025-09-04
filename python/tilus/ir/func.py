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
from __future__ import annotations as _

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from hidet.ir.expr import Expr, Var

from tilus.ir.node import IRNode
from tilus.ir.stmt import Stmt
from tilus.ir.utils import frozendict


@dataclass(frozen=True, eq=False)
class Analysis:
    divisibility: frozendict[Var, int]
    lower_bound: frozendict[Var, int]
    upper_bound: frozendict[Var, int]

    @staticmethod
    def create(
        divisibility: Mapping[Var, int], lower_bound: Mapping[Var, int], upper_bound: Mapping[Var, int]
    ) -> Analysis:
        return Analysis(frozendict(divisibility), frozendict(lower_bound), frozendict(upper_bound))

    @staticmethod
    def empty() -> Analysis:
        return Analysis(frozendict(), frozendict(), frozendict())


@dataclass(frozen=True)
class Metadata:
    num_blocks: tuple[Expr, Expr, Expr]
    num_blocks_per_cluster: tuple[int, int, int]
    block_indices: tuple[Var, Var, Var]
    num_warps: int
    param2divisibility: frozendict[Var, int]
    analysis: Optional[Analysis]

    @staticmethod
    def create(
        num_blocks: Sequence[Expr],
        num_blocks_per_cluster: Sequence[int],
        block_indices: Sequence[Var],
        num_warps: int,
        divisibility: Optional[Mapping[Var, int]] = None,
        analysis: Optional[Analysis] = None,
    ) -> Metadata:
        assert len(num_blocks) == 3 and len(block_indices) == 3 and len(num_blocks_per_cluster) == 3

        return Metadata(
            num_blocks=(num_blocks[0], num_blocks[1], num_blocks[2]),
            num_blocks_per_cluster=(num_blocks_per_cluster[0], num_blocks_per_cluster[1], num_blocks_per_cluster[2]),
            block_indices=(block_indices[0], block_indices[1], block_indices[2]),
            num_warps=num_warps,
            param2divisibility=frozendict(divisibility) if divisibility else frozendict(),
            analysis=analysis,
        )

    def with_analysis(self, analysis: Optional[Analysis]) -> Metadata:
        return dataclasses.replace(self, analysis=analysis)

    def with_num_blocks(self, num_blocks: tuple[Expr, Expr, Expr]) -> Metadata:
        return dataclasses.replace(self, num_blocks=num_blocks)


@dataclass(frozen=True, eq=False)
class Function(IRNode):
    name: str
    params: tuple[Var, ...]
    body: Stmt
    metadata: Metadata

    @staticmethod
    def create(
        name: str,
        params: Sequence[Var],
        body: Stmt,
        metadata: Metadata,
    ) -> Function:
        return Function(
            name,
            tuple(params),
            body,
            metadata,
        )

    def with_body(self, new_body: Stmt) -> Function:
        return dataclasses.replace(self, body=new_body)

    def with_name(self, new_name: str) -> Function:
        return dataclasses.replace(self, name=new_name)

    def with_metadata(self, new_metadata: Metadata) -> Function:
        return dataclasses.replace(self, metadata=new_metadata)
