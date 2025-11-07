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
from typing import Any, Dict, List, Set, Tuple, Union

from hidet.ir import BaseType
from hidet.ir.expr import Expr, Var
from hidet.ir.tools import IRPrinter as HidetIRPrinter
from hidet.utils.doc import Doc, NewLine, Text, doc_join

from tilus.extensions.hidet.utils.doc import doc_comment, doc_join_lines, doc_strip_parentheses
from tilus.ir.func import Analysis, Function, Metadata
from tilus.ir.functors import IRFunctor
from tilus.ir.inst import Instruction, InstructionConfig
from tilus.ir.layout import RegisterLayout
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    BreakStmt,
    DeclareStmt,
    EvaluateStmt,
    ForStmt,
    IfStmt,
    InstStmt,
    LetStmt,
    ReturnStmt,
    SeqStmt,
    TensorItemPtrStmt,
    TensorItemValueStmt,
    ThreadGroupStmt,
    WhileStmt,
)
from tilus.ir.tensor import (
    GlobalLayout,
    GlobalTensor,
    RegisterTensor,
    SharedLayout,
    SharedTensor,
    Tensor,
    TMemoryTensor,
)


class IRPrinter(IRFunctor):
    def __init__(self) -> None:
        super().__init__()
        self.printer = HidetIRPrinter()
        self.tensor2name: Dict[Tensor, str] = {}
        self.var2name: Dict[Var, str] = {}
        self.comment2key: Dict[str, str] = {}
        self.key2comment: Dict[str, str] = {}
        self.keys: Set[str] = set()

        self.in_function: bool = False
        self.shared_count: int = 0
        self.register_count: int = 0
        self.global_count: int = 0
        self.tmem_count: int = 0
        self.var_count: int = 0
        self.ptr_count: int = 0

    def set_var_name(self, var: Var, name: str) -> None:
        self.var2name[var] = name
        self.printer.namer.obj_name[var] = name

    def add_key_comment(self, key_hint: str, comment: str | Doc) -> Doc:
        comment_doc: Doc = Text(comment) if isinstance(comment, str) else comment
        comment_str: str = str(comment_doc)
        if not self.in_function:
            return comment_doc

        if comment_str in self.comment2key:
            return Text(self.comment2key[comment_str])
        i = 0
        while True:
            key = key_hint + "_" + str(i)
            if key not in self.keys:
                self.keys.add(key)
                self.comment2key[comment_str] = key
                self.key2comment[key] = comment_str
                return Text(key)
            i += 1

    def get_tensor_type(self, tensor: Tensor) -> Doc:
        if isinstance(tensor, RegisterTensor):
            doc = Text("register, ")
            doc += self.printer(tensor.dtype) + "[" + self.visit(tensor.shape) + "]"
            if tensor.optional_layout is not None:
                doc += ", local_size={}".format(tensor.layout.local_size)
                doc += ", {}".format(self.visit(tensor.layout))
            return doc
        elif isinstance(tensor, SharedTensor):
            doc = Text("shared, ")
            doc += self.printer(tensor.dtype) + "[" + self.visit(tensor.shape) + "]"
            if tensor.optional_layout is not None:
                doc += ", size={}".format(tensor.layout.size)
                doc += ", {}".format(self.visit(tensor.layout))
            return doc
        elif isinstance(tensor, GlobalTensor):
            doc = Text("global, ")
            doc += self.printer(tensor.dtype) + "[" + self.visit(tensor.shape) + "], "
            doc += "size={}".format(self.printer(tensor.layout.size))
            doc += ", {}".format(self.visit(tensor.layout))
            return doc
        elif isinstance(tensor, TMemoryTensor):
            doc = Text("tensor_memory, ")
            doc += self.printer(tensor.dtype) + "[" + self.visit(tensor.shape) + "]"
            return doc
        else:
            raise NotImplementedError()

    def visit_list(self, lst: List) -> Doc:
        return doc_join([doc_strip_parentheses(self.visit(node)) for node in lst], ", ")

    def visit_tuple(self, lst: Tuple) -> Doc:
        return doc_join([doc_strip_parentheses(self.visit(node)) for node in lst], ", ")

    def visit_dict(self, node: Dict) -> Doc:
        items = []
        for key, value in node.items():
            key_doc = self.visit(key)
            value_doc = self.visit(value)
            if isinstance(value, list):
                value_doc = "[" + value_doc + "]"
            elif isinstance(value, tuple):
                value_doc = "[" + value_doc + "]"
            elif isinstance(value, dict):
                value_doc = "{" + value_doc + "}"
            elif isinstance(value, frozenset):
                value_doc = "{" + value_doc + "}"
            items.append(key_doc + ": " + doc_strip_parentheses(value_doc))
        return doc_join(items, ", ")

    def visit_frozenset(self, node: frozenset) -> Doc:
        return doc_join([doc_strip_parentheses(self.visit(node)) for node in node], ", ")

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> Doc:
        if isinstance(node, str):
            return Text(repr(node))
        else:
            return Text(str(node))

    def visit_Expr(self, expr: Expr) -> Doc:
        if isinstance(expr, Var):
            if expr not in self.var2name:
                if expr.type.is_pointer():
                    symbol = "%p"
                    count = self.ptr_count
                    self.ptr_count += 1
                else:
                    symbol = "%v"
                    count = self.var_count
                    self.var_count += 1
                name = symbol + str(count)
                self.set_var_name(expr, name)
            return Text(self.var2name[expr])
        else:
            return self.printer(expr)

    def visit_BaseType(self, tp: BaseType) -> Doc:
        return self.printer(tp)

    def visit_Program(self, prog: Program) -> Doc:
        doc = Doc()

        for func in prog.functions.values():
            doc += self.visit(func) + NewLine()

        return doc

    def visit_Analysis(self, analysis: Analysis) -> Doc:
        doc = Doc()
        if analysis.divisibility:
            doc += NewLine() + "divisibility = {" + self.visit(analysis.divisibility) + "}"
        if analysis.lower_bound:
            doc += NewLine() + "lower_bound = {" + self.visit(analysis.lower_bound) + "}"
        if analysis.upper_bound:
            doc += NewLine() + "upper_bound = {" + self.visit(analysis.upper_bound) + "}"
        return doc

    def visit_FuncMetadata(self, metadata: Metadata) -> Doc:
        doc = Doc()
        doc += NewLine() + "grid_blocks = [" + self.visit(metadata.grid_blocks) + "]"
        doc += NewLine() + "cluster_blocks= [" + self.visit(metadata.cluster_blocks) + "]"
        doc += NewLine() + "num_warps = " + self.visit(metadata.num_warps)
        if metadata.param2divisibility:
            doc += NewLine() + "param_divisibility = {" + self.visit(metadata.param2divisibility) + "}"
        if metadata.analysis:
            doc += NewLine() + "analysis = {"
            doc += self.visit_Analysis(metadata.analysis).indent(4)
            doc += NewLine() + "}"
        doc = doc_comment(doc, "# ")
        doc += NewLine()
        return doc

    def visit_Function(self, func: Function) -> Doc:
        for block_index in func.metadata.block_indices:
            # we use blockIdx.x, blockIdx.y, blockIdx.z as default block indices name
            self.set_var_name(block_index, block_index.name)

        # head doc
        doc = doc_join_lines(
            seq=[self.visit(p) + ": " + self.printer(p.type) for p in func.params],
            left="def " + func.name + "(",
            right="):",
        )

        # first visit body to make sure the variable names are set in their definition order
        self.in_function = True
        body_doc = self.visit(func.body)
        self.in_function = False

        # metadata doc
        doc += self.visit_FuncMetadata(func.metadata).indent(4)

        # body doc
        doc += body_doc.indent(4)

        # comment doc
        key_comment_items = sorted([(v, k) for k, v in self.comment2key.items()], key=lambda kv: kv[0])
        doc += NewLine()
        doc += NewLine() + doc_comment(
            doc_join(seq=[key + ": " + comment for key, comment in key_comment_items], sep=NewLine()), "# "
        )

        return doc

    def visit_InstStmt(self, stmt: InstStmt) -> Doc:
        return NewLine() + self.visit(stmt.inst)

    def visit_SeqStmt(self, stmt: SeqStmt) -> Doc:
        doc = Doc()
        for sub_stmt in stmt.seq:
            doc += self.visit(sub_stmt)
        return doc

    def visit_ForStmt(self, stmt: ForStmt) -> Doc:
        doc = Doc()
        if stmt.unroll_factor:
            if stmt.unroll_factor == -1:
                doc += NewLine() + "#pragma unroll"
            else:
                doc += NewLine() + "#pragma unroll {}".format(stmt.unroll_factor)
        doc += NewLine() + Text("for ") + self.visit(stmt.iter_var) + " in range(" + self.visit(stmt.extent) + "):"
        doc += self.visit(stmt.body).indent(4)
        return doc

    def visit_ThreadGroupStmt(self, stmt: ThreadGroupStmt) -> Doc:
        head_doc = (
            NewLine()
            + Text("with thread_group(group_index=")
            + self.visit(stmt.group_index)
            + ", group_size="
            + self.visit(stmt.group_size)
            + "):"
        )
        body_doc = self.visit(stmt.body)
        doc = head_doc + body_doc.indent(4)
        return doc

    def visit_IfStmt(self, stmt: IfStmt) -> Doc:
        doc = NewLine() + Text("if ") + self.visit(stmt.cond) + ":"
        doc += self.visit(stmt.then_body).indent(4)
        if stmt.else_body is not None:
            doc += NewLine() + Text("else:")
            doc += self.visit(stmt.else_body).indent(4)
        return doc

    def visit_WhileStmt(self, stmt: WhileStmt) -> Doc:
        doc = NewLine() + Text("while ") + self.visit(stmt.cond) + ":"
        doc += self.visit(stmt.body).indent(4)
        return doc

    def visit_BreakStmt(self, stmt: BreakStmt) -> Doc:
        return NewLine() + Text("break")

    def visit_ReturnStmt(self, stmt: ReturnStmt) -> Any:
        return NewLine() + Text("return")

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Doc:
        ret = NewLine() + Text("declare ") + self.visit(stmt.var) + ": " + self.printer(stmt.var.type)
        if stmt.init is not None:
            ret += " = " + self.visit(stmt.init)
        return ret

    def visit_EvaluateStmt(self, stmt: EvaluateStmt) -> Doc:
        doc = Doc()
        doc += NewLine() + self.visit(stmt.expr)
        return doc

    def visit_LetStmt(self, stmt: LetStmt) -> Doc:
        doc = Doc()
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            doc += (
                NewLine()
                + Text("let ")
                + self.visit(bind_var)
                + ": "
                + self.printer(bind_var.type)
                + " = "
                + self.visit(bind_value)
            )
        doc += self.visit(stmt.body)
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt) -> Doc:
        return NewLine() + self.visit(stmt.var) + " = " + self.visit(stmt.value)

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> Doc:
        doc = Doc()
        doc += (
            NewLine()
            + self.visit(stmt.ptr_var)
            + ": "
            + self.printer(stmt.ptr_var.type)
            + " = "
            + self.visit(stmt.tensor)
            + ".item_ptr(space='"
            + stmt.space
            + "')"
        )
        return doc

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> Any:
        doc = Doc()
        doc += (
            NewLine()
            + self.visit(stmt.var)
            + ": "
            + self.printer(stmt.var.type)
            + " = "
            + self.visit(stmt.tensor)
            + ".item()"
        )
        return doc

    def visit_Instruction(self, inst: Instruction) -> Doc:
        doc = Doc()
        if inst.output is not None:
            doc += self.visit(inst.output) + " = "
        inst_name = inst.__class__.__name__.removesuffix("Inst")
        doc += inst_name + "("

        items = []
        if len(inst.inputs):
            items.append(self.visit(inst.inputs))
        for k, v in inst.attributes.items():
            if v is None:
                continue
            if k == "axes" and isinstance(v, tuple) and all(isinstance(vv, Var) for vv in v):
                axes = v
                for i, axis in enumerate(axes):
                    self.set_var_name(axis, f"u{i}")
            v_doc = doc_strip_parentheses(self.visit(v))
            if isinstance(v, (list, tuple)):
                v_doc = "[" + v_doc + "]"
            elif isinstance(v, dict):
                v_doc = "{" + v_doc + "}"
            items.append("{}={}".format(k, v_doc))
        items = [str(item) for item in items]
        if sum(len(item) for item in items) >= 80:
            item_body = Doc()
            for i, item in enumerate(items):
                item_body += NewLine() + Text(item)
                if i != len(items) - 1:
                    item_body += ","
            item_body = item_body.indent(4)
            item_body += NewLine()
        else:
            item_body = doc_join(items, ", ")
        doc += item_body
        doc += ")"
        if inst.output is not None:
            doc += "  # " + self.get_tensor_type(inst.output)
        return doc

    def visit_InstructionConfig(self, inst_config: InstructionConfig) -> Any:
        name2value = inst_config.__dict__
        doc = Doc()
        doc += Text(inst_config.__class__.__name__)
        doc += "("
        items = []
        for name, value in name2value.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                value = "[" + self.visit(value) + "]"
            elif isinstance(value, dict):
                value = "{" + self.visit(value) + "}"
            items.append(Text(name) + "=" + self.visit(value))
        doc += doc_join(items, ", ")
        doc += ")"
        return doc

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> Doc:
        if tensor not in self.tensor2name:
            self.tensor2name[tensor] = "%r" + str(self.register_count)
            self.register_count += 1
        return Text(self.tensor2name[tensor])

    def visit_SharedTensor(self, tensor: SharedTensor) -> Doc:
        if tensor not in self.tensor2name:
            self.tensor2name[tensor] = "%s" + str(self.shared_count)
            self.shared_count += 1
        return Text(self.tensor2name[tensor])

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> Doc:
        if tensor not in self.tensor2name:
            self.tensor2name[tensor] = "%g" + str(self.global_count)
            self.global_count += 1
        return Text(self.tensor2name[tensor])

    def visit_TensorMemoryTensor(self, tensor: TMemoryTensor) -> Doc:
        if tensor not in self.tensor2name:
            self.tensor2name[tensor] = "%t" + str(self.tmem_count)
            self.tmem_count += 1
        return Text(self.tensor2name[tensor])

    def visit_RegisterLayout(self, layout: RegisterLayout) -> Doc:
        items = [
            "shape=[" + self(layout.shape) + "]",
            "mode_shape=[" + self(layout.mode_shape) + "]",
            "spatial_modes=[" + self(layout.spatial_modes) + "]",
            "local_modes=[" + self(layout.local_modes) + "]",
        ]
        doc = Text("RegisterLayout(") + doc_join(items, ", ") + ")"
        return self.add_key_comment("layout", doc)

    def visit_SharedLayout(self, node: SharedLayout) -> Doc:
        for i, axis in enumerate(node.axes):
            self.set_var_name(axis, "u" + str(i))
        items = [
            "shape=[" + self(node.shape) + "]",
            "axes=[" + self(node.axes) + "]",
            "offset=" + self(node.offset),
        ]
        doc = Text("SharedLayout(") + doc_join(items, ", ") + ")"
        return self.add_key_comment("shared_layout", doc)

    def visit_GlobalLayout(self, node: GlobalLayout) -> Doc:
        for i, axis in enumerate(node.axes):
            self.set_var_name(axis, "u" + str(i))
        items = [
            "shape=[" + self(node.shape) + "]",
            "axes=[" + self(node.axes) + "]",
            "offset=" + self(node.offset),
        ]
        doc = Text("GlobalLayout(") + doc_join(items, ", ") + ")"
        return self.add_key_comment("global_layout", doc)


class PrintContext:
    _printer_stack: list[IRPrinter] = []

    def __init__(self) -> None:
        self.printer = IRPrinter()

    def __enter__(self):
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def enter(self):
        """
        Enter the print context.
        """
        self._printer_stack.append(self.printer)

    def exit(self):
        """
        Exit the print context.
        """
        if not self._printer_stack or self._printer_stack[-1] is not self.printer:
            raise RuntimeError()
        self._printer_stack.pop()

    @staticmethod
    def current() -> IRPrinter:
        """
        Get the current print context.
        """
        if not PrintContext._printer_stack:
            return IRPrinter()
        return PrintContext._printer_stack[-1]


def print_context() -> PrintContext:
    """
    Create a new print context for printing IR nodes.
    """
    return PrintContext()


def use_standalone_printer(decorated):
    """
    A decorator to use a standalone IRPrinter instance for the decorated function.
    This is useful when you want to print IR nodes without affecting the global print context.
    """

    def wrapper(*args, **kwargs):
        with print_context():
            return decorated(*args, **kwargs)

    return wrapper
