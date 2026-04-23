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
from typing import Dict, Sequence

from tilus.hidet.ir.expr import Expr, TensorElement, Var, convert, tensor_element
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.functors import IRRewriter
from tilus.hidet.ir.module import IRModule
from tilus.hidet.ir.stmt import BufferStoreStmt, DeclareStmt
from tilus.hidet.ir.tools import TypeInfer, simplify
from tilus.hidet.ir.type import (
    FuncType,
    PointerType,
    TensorPointerType,
    TensorType,
    func_type,
    tensor_pointer_type,
    tensor_type,
)
from tilus.hidet.ir.utils.call_graph import CallGraph, CallGraphNode
from tilus.hidet.transforms.base import Pass
from tilus.hidet.utils import prod


def _row_major_index(shape: Sequence[Expr], indices: Sequence[Expr]) -> Expr:
    assert len(shape) == len(indices)
    if len(shape) == 0:
        return convert(0)
    index: Expr = convert(indices[0])
    for dim in range(1, len(shape)):
        index = index * shape[dim] + indices[dim]
    return index


class FlattenTensorAccessRewriter(IRRewriter):
    # flatten all high-dimension tensor access
    # A = int[3, 4]
    #   TensorElement:  A[2, 1]     ==> A[2 * 4 + 1]
    # BufferStoreStmt:  A[2, 1] = 3 ==> A[2 * 4 + 1] = 3
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()
        self.func2func_type: Dict[str, FuncType] = {}

    def visit_Var(self, v: Var):
        if isinstance(v.type, FuncType):
            if v.name in self.func2func_type:
                func_ty = self.func2func_type[v.name]
                if func_ty is not v.type:
                    new_var = Var(v.name, func_ty)
                    self.memo[v] = new_var
                    return new_var
        return super().visit_Var(v)

    def visit_Function(self, func: Function):
        for var in func.params:
            if isinstance(var.type, TensorType):
                size = simplify(prod(var.type.shape))
                self.memo[var] = Var(var.name, tensor_pointer_type(var.type.dtype, [size]))
            elif isinstance(var.type, TensorPointerType):
                size = simplify(prod(var.type.tensor_type.shape))
                self.memo[var] = Var(var.name, tensor_pointer_type(var.type.tensor_type.dtype, [size]))
        body = self(func.body)
        params = [self(p) for p in func.params]
        if body is func.body and all(p is p1 for p, p1 in zip(params, func.params)):
            return func
        else:
            new_func = Function(func.name, params, body, func.ret_type, kind=func.kind, attrs=func.attrs)
            param_types = [p.type for p in params]
            self.func2func_type[func.name] = func_type(param_types, func.ret_type)
            return new_func

    def get_shape(self, e) -> Sequence[Expr]:
        e_type = self.type_infer(e)

        if isinstance(e_type, TensorType):
            return e.type.shape
        elif isinstance(e_type, TensorPointerType):
            return e.type.tensor_type.shape
        elif isinstance(e_type, PointerType):
            return [convert(0)]
        else:
            raise ValueError("Can not infer shape from '{}' (expression {})".format(type(e), e))

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.var.type, TensorType):
            size = simplify(prod(stmt.var.type.shape))
            var = Var(stmt.var.name, tensor_type(stmt.var.type.dtype, [size]))
            self.memo[stmt.var] = var
            init = self(stmt.init) if stmt.init is not None else None
            return DeclareStmt.create(var, init, is_static=stmt.is_static, scope=stmt.scope)
        elif isinstance(stmt.var.type, TensorPointerType):
            size = simplify(prod(stmt.var.type.tensor_type.shape))
            var = Var(stmt.var.name, tensor_pointer_type(stmt.var.type.tensor_type.dtype, [size]))
            self.memo[stmt.var] = var
            init = self(stmt.init) if stmt.init is not None else None
            return DeclareStmt.create(var, init, is_static=stmt.is_static, scope=stmt.scope)
        else:
            return IRRewriter.visit_DeclareStmt(self, stmt)

    def visit_TensorElement(self, e: TensorElement):
        var = self(e.base)
        indices = [self(i) for i in e.indices]
        shape = self.get_shape(e.base)
        if len(indices) != len(shape):
            raise ValueError(
                "Access {}-d tensor {} named {} with {}-d indices {}".format(
                    len(shape), list(shape), var.name, len(indices), list(indices)
                )
            )
        global_index = _row_major_index(shape, indices)
        return tensor_element(var, (global_index,))

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        var = self(stmt.buf)
        indices = [self(i) for i in stmt.indices]
        value = self(stmt.value)
        shape = self.get_shape(stmt.buf)
        if len(shape) != len(indices):
            raise ValueError(
                "Access {}-d tensor {}{} with {}-d indices {}".format(
                    len(shape), var.name, list(shape), len(indices), list(indices)
                )
            )
        global_index = _row_major_index(shape, indices)
        return BufferStoreStmt.create(var, [global_index], value)


class FlattenTensorIndexPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        flatten_index = FlattenTensorAccessRewriter()

        new_funcs = {}
        call_graph = CallGraph(ir_module, allow_missing=True)
        for node in call_graph.reversed_order:
            assert isinstance(node, CallGraphNode)
            name = node.func.name
            func = node.func
            new_funcs[name] = flatten_index(func)

        if all(new_funcs[name] is ir_module.functions[name] for name in new_funcs):
            return ir_module
        return ir_module.with_functions(new_funcs, ir_module.global_vars)


def flatten_tensor_index_pass():
    return FlattenTensorIndexPass()
