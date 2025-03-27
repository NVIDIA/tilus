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
"""
Some expressions in a function should be expressions only over parameters:
1. The AllocateGlobal instruction requires the size to be only over parameters if they are not constant.
2. The num_blocks should also be only over parameters if they are not constant.

However, we allow that the user to write a program that define some intermediate variables to compute the size or
num_blocks. This pass will canonicalize the expressions in the function to be only over parameters if they are not.
This pass requires that we have converted all DeclareStmt that is only assigned once to LetStmt. Thus, the
`declare_to_let` pass that performs this transformation should be run before this pass.

The algorithm is simple:
1. we record the variable to expression binding in LetStmt
2. for each expression that should be parameter-only, we replace the variables in the expression that are not parameters
   with the let binding expression, until the expression is parameter-only, or the expression contains variables that
   can not be resolved to parameters (e.g., those are loop iterator variables or variables that have been assigned more
   than once).
"""

from hidet.ir.expr import Expr, Var
from hidet.ir.tools import collect
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.instructions import AllocateGlobalInst
from tilus.ir.layout import GlobalLayout
from tilus.ir.stmt import LetStmt, Stmt
from tilus.ir.tensor import GlobalTensor
from tilus.transforms.base import Pass


class LowerParamOnlyExprRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.params: tuple[Var, ...] = ()
        self.var2expr: dict[Var, Expr] = {}

    def lower_param_only_param(self, expr: Expr, allow_vars: tuple[Var, ...] = ()) -> Expr:
        rewrite_map = self.var2expr
        allow_vars = self.params + allow_vars
        while True:
            used_vars = collect(expr, [Var])
            if all(v in allow_vars for v in used_vars):
                # we are good now
                return expr
            else:
                if not any(v in self.var2expr for v in used_vars if v not in allow_vars):
                    # we can not resolve some variables
                    illegal_vars = [v for v in used_vars if v not in self.var2expr and v not in allow_vars]
                    raise ValueError(
                        "Used variables {} that is not parameter in parameter-only expression.".format(illegal_vars)
                    )
                expr = rewrite(expr, rewrite_map)

    def visit_Function(self, func: Function) -> Function:
        self.params = func.params
        body = self.visit(func.body)
        num_blocks = func.metadata.num_blocks
        num_blocks = (
            self.lower_param_only_param(num_blocks[0]),
            self.lower_param_only_param(num_blocks[1]),
            self.lower_param_only_param(num_blocks[2]),
        )
        if body is func.body and all(e is t for e, t in zip(func.metadata.num_blocks, num_blocks)):
            return func
        else:
            return func.with_metadata(func.metadata.with_num_blocks(num_blocks)).with_body(body)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.var2expr[bind_var] = bind_value
        body = self.visit(stmt.body)

        if body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, stmt.bind_values, body)

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst) -> AllocateGlobalInst:
        global_tensor: GlobalTensor = inst.global_output
        global_layout: GlobalLayout = global_tensor.layout

        size = self.lower_param_only_param(global_layout.size)
        shape = tuple(self.lower_param_only_param(s) for s in global_layout.shape)
        offset = self.lower_param_only_param(global_layout.offset, allow_vars=global_layout.axes)
        if (
            size is global_layout.size
            and all(s is t for s, t in zip(global_layout.shape, shape))
            and offset is global_layout.offset
        ):
            return inst
        else:
            global_layout = GlobalLayout(shape=shape, size=size, axes=global_layout.axes, offset=offset)
            new_global_tensor = global_tensor.with_layout(global_layout)
            self.memo[global_tensor] = new_global_tensor
            return inst.with_output(global_output=new_global_tensor)


class LowerParamOnlyExprPass(Pass):
    def process_function(self, func: Function) -> Function:
        rewriter = LowerParamOnlyExprRewriter()
        return rewriter(func)


def lower_param_only_expr_pass() -> Pass:
    return LowerParamOnlyExprPass()
