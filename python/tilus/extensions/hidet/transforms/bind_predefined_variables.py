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
"""Bind the pre-defined variables (that are block-invariant) to their values at the beginning of the function."""

from hidet.ir.expr import Expr, Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt
from hidet.transforms.base import FunctionPass

from tilus.extensions.hidet.ir.primitives.cuda.cluster import (
    block_id_in_cluster,
    block_rank_in_cluster,
    cluster_blocks,
    cluster_id_in_grid,
    cluster_shape,
)
from tilus.extensions.hidet.ir.primitives.cuda.vars import (
    clusterBlockIdx,
    clusterBlockRank,
    clusterDim,
    clusterIdx,
    clusterSize,
)


class BindPredefinedVariablesRewriter(IRRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.var2value: dict[Var, Expr] = {}
        self.used_vars: list[Var] = []

    def visit_Var(self, e):
        if e in self.var2value and e not in self.used_vars:
            self.used_vars.append(e)
        return super().visit_Var(e)

    def visit_Function(self, func):
        # fill the var2value mapping
        self.var2value = {
            clusterBlockIdx.x: block_id_in_cluster("x"),
            clusterBlockIdx.y: block_id_in_cluster("y"),
            clusterBlockIdx.z: block_id_in_cluster("z"),
            clusterSize: cluster_blocks(),
            clusterDim.x: cluster_shape("x"),
            clusterDim.y: cluster_shape("y"),
            clusterDim.z: cluster_shape("z"),
            clusterBlockRank: block_rank_in_cluster(),
            clusterIdx.x: cluster_id_in_grid("x"),
            clusterIdx.y: cluster_id_in_grid("y"),
            clusterIdx.z: cluster_id_in_grid("z"),
        }

        body = self.visit(func.body)

        if self.used_vars:
            # bind the used predefined variables at the beginning of the function
            bind_vars = []
            bind_values = []
            for var in self.used_vars:
                bind_vars.append(var)
                bind_values.append(self.var2value[var])
            body = LetStmt(bind_vars, bind_values, body)
            return Function(func.name, func.params, body, func.ret_type, kind=func.kind, attrs=func.attrs)
        else:
            # no predefined variable is used
            return func


class BindPredefinedVariablesPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = BindPredefinedVariablesRewriter()
        return rewriter.visit(func)


def bind_predefined_variables_pass() -> FunctionPass:
    return BindPredefinedVariablesPass()
