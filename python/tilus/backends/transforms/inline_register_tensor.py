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
from hidet.ir import ForStmt, Function
from hidet.ir.dtypes import int32
from hidet.ir.expr import Address, Expr, TensorElement, Var
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.primitives.cuda.vars import blockDim, blockIdx, gridDim, threadIdx
from hidet.ir.stmt import BufferStoreStmt, DeclareScope, DeclareStmt, SeqStmt, Stmt
from hidet.ir.tools import collect, rewrite
from hidet.ir.type import TensorType
from hidet.transforms.base import FunctionPass, Pass

"""
If there is only one assignment for a register tensor, and its value is the indexing of another register tensor,
we can inline the register tensor to avoid the extra copy. For example:
```
declare rhs[16]    # register scope
declare lhs[128]    # register scope
for i in range(128):
    lhs[i] = rhs[expr(i)] # only one assignment, so it's safe to inline
```
all usage of lhs[p] will be rewritten to rhs[expr(p)]

There some conditions to inline:
1. both lhs and rhs are constant register tensors that are assigned only once
2. the assignment for lhs has form `lhs[i] = rhs[expr(i)]`
3. the lhs is only used in TensorIndex statements
4. the expr(i) only contains constants, thread-constant variables (threadIdx.x, threadIdx.y, etc.), and i.
"""


class TensorMapping:
    def __init__(self, rhs, axis, index_expr):
        self.rhs: Var = rhs
        self.axis: Var = axis
        self.index_expr: Expr = index_expr


class RegisterTensorCollector(IRVisitor):
    """
    Collect all register tensors:
    1. it is 1-d tensor
    2. it is only used in BufferStoreStmt (as lhs) or TensorElement (as base) statements
    """

    def __init__(self):
        super().__init__()
        # the 1-d register tensors that is only used in BufferStoreStmt (as lhs) or TensorElement (as base)
        self.register_tensors: list[Var] = []

    def visit_Var(self, e: Var) -> None:
        if e in self.register_tensors:
            self.register_tensors.remove(e)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        if stmt.scope != DeclareScope.Register:
            return super().visit_DeclareStmt(stmt)
        var_type = stmt.var.type
        if not isinstance(var_type, TensorType):
            return super().visit_DeclareStmt(stmt)
        if len(var_type.shape) != 1:
            # we only support 1D register tensors, since tilus only generate 1D register tensors
            return super().visit_DeclareStmt(stmt)
        if stmt.init is not None:
            return super().visit_DeclareStmt(stmt)
        self.register_tensors.append(stmt.var)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt) -> None:
        self.visit(stmt.value)

    def visit_TensorElement(self, e: TensorElement) -> None:
        self.visit(e.indices)

    def visit_Address(self, e: Address) -> None:
        # we should exclude buf used like ~buf[xxx]
        if isinstance(e.expr, TensorElement):
            base = e.expr.base
            if isinstance(base, Var) and base in self.register_tensors:
                self.register_tensors.remove(base)
        super().visit_Address(e)


class RegisterTensorFilter(IRVisitor):
    """
    Filter out the register tensors that are assigned multiple times
    """

    def __init__(self, register_tensors: list[Var]):
        super().__init__()
        self.register_tensors = register_tensors
        self.num_assignments: dict[Var, int] = {var: 0 for var in register_tensors}

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt) -> None:
        if stmt.buf in self.register_tensors:
            assert isinstance(stmt.buf, Var)
            self.num_assignments[stmt.buf] += 1
            if self.num_assignments[stmt.buf] > 1:
                self.register_tensors.remove(stmt.buf)


class RegisterMappingCollector(IRVisitor):
    def __init__(self, register_tensors: list[Var]):
        super().__init__()
        self.register_tensors = register_tensors
        self.tensor_mapping: dict[Var, TensorMapping] = {}

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt) -> None:
        if stmt.buf not in self.register_tensors:
            return
        if not isinstance(stmt.value, TensorElement):
            return
        if len(stmt.indices) != 1 or not isinstance(stmt.indices[0], Var):  # type: ignore[arg-type]
            return
        te = stmt.value
        axis = stmt.indices[0]
        if len(te.indices) != 1:
            return
        if te.base not in self.register_tensors:
            return
        used_vars = collect(te.indices[0], Var)
        allowed_vars = [
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
            gridDim.z,  # type: ignore
            axis,
        ]
        for var in used_vars:
            if var not in allowed_vars:
                return
        self.tensor_mapping[stmt.buf] = TensorMapping(rhs=te.base, axis=axis, index_expr=te.indices[0])  # type: ignore


class InlineRegisterTensorRewriter(IRRewriter):
    def __init__(self, tensor_mapping: dict[Var, TensorMapping]):
        super().__init__()
        self.tensor_mapping: dict[Var, TensorMapping] = tensor_mapping

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Stmt:
        if stmt.var in self.tensor_mapping:
            # remove the register tensor declaration
            return SeqStmt([])
        return super().visit_DeclareStmt(stmt)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt) -> Stmt:
        if stmt.buf in self.tensor_mapping:
            return SeqStmt([])
        return super().visit_BufferStoreStmt(stmt)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        stmt = super().visit_ForStmt(stmt)
        if isinstance(stmt, ForStmt) and isinstance(stmt.body, SeqStmt) and len(stmt.body.seq) == 0:
            return SeqStmt([])
        return stmt

    def visit_SeqStmt(self, stmt: SeqStmt) -> Stmt:
        seq: list[Stmt] = []
        for s in stmt.seq:
            ss = self.visit(s)
            if isinstance(ss, SeqStmt):
                seq.extend(ss.seq)
            else:
                seq.append(ss)
        if len(seq) == len(stmt.seq) and all(s1 is s2 for s1, s2 in zip(seq, stmt.seq)):
            return stmt
        return SeqStmt(seq)

    def visit_TensorElement(self, e: TensorElement) -> Expr:
        if e.base in self.tensor_mapping:
            assert isinstance(e.base, Var)
            mapping = self.tensor_mapping[e.base]
            assert len(e.indices) == 1
            index = e.indices[0]
            index_var = Var("idx", int32)
            self.append_prologue_stmt(
                DeclareStmt(
                    var=index_var,
                    init=index,
                )
            )
            new_index = rewrite(mapping.index_expr, {mapping.axis: index_var})
            return TensorElement(mapping.rhs, (new_index,))
        return super().visit_TensorElement(e)


class InlineRegisterTensorPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        # collect all register tensors
        collector = RegisterTensorCollector()
        collector.visit(func.body)
        register_tensors = collector.register_tensors

        # filter out the register tensors that are assigned multiple times
        filter = RegisterTensorFilter(register_tensors)
        filter.visit(func.body)

        # collect the mapping for each register tensor
        collector = RegisterMappingCollector(register_tensors)
        collector.visit(func.body)
        tensor_mapping = collector.tensor_mapping

        # inline the register tensors
        rewriter = InlineRegisterTensorRewriter(tensor_mapping)
        return rewriter(func)


def inline_register_tensor_pass() -> Pass:
    return InlineRegisterTensorPass()
