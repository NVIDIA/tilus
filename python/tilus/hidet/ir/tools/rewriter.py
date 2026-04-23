# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Mapping, Union

from tilus.hidet.ir.expr import Let, Var
from tilus.hidet.ir.functors import IRRewriter
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.stmt import DeclareStmt, ForStmt, LetStmt


class MapBasedRewriter(IRRewriter):
    def __init__(self, rmap):
        super().__init__()
        self.memo.update(rmap)


class CloneRewriter(IRRewriter):
    """
    A rewriter that will create a new var for each statement/expr that will declare vars
    """

    def __init__(self, remap: Dict[Node, Node]):
        super().__init__()
        self.memo.update(remap)

    def process_var(self, v: Var):
        visited_v = self.visit(v)
        if visited_v is v:
            new_var = Var(v.name, type=v.type)
        else:
            new_var = visited_v
        self.memo[v] = new_var
        return new_var

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self.process_var(stmt.loop_var)
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        return ForStmt.create(loop_var, extent, body, attr=stmt.attr)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = [self.process_var(v) for v in stmt.bind_vars]
        bind_values = [self.visit(bind_value) for bind_value in stmt.bind_values]
        body = self.visit(stmt.body)
        return LetStmt.create(bind_vars, bind_values, body)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = self.process_var(stmt.var)
        init = self.visit(stmt.init) if stmt.init is not None else None
        return DeclareStmt.create(v, init, stmt.is_static, stmt.scope)

    def visit_Let(self, e: Let):
        v = self.process_var(e.var)
        value = self.visit(e.value)
        body = self.visit(e.body)
        return Let(v, value, body)


def rewrite(node: Union[Node, tuple, list, dict], rewrite_map: Mapping[Node, Node], clone_internal_var=False):
    assert isinstance(rewrite_map, dict)
    if clone_internal_var:
        rewriter = CloneRewriter(rewrite_map)
    else:
        rewriter = MapBasedRewriter(rewrite_map)
    return rewriter.rewrite(node)
