from typing import List, Tuple, Dict, Union, Set, Any

from hidet.ir import BaseType
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.ir.expr import Expr
from tilus.ir.layout import Layout
from tilus.ir.func import Function
from tilus.ir.prog import Program
from tilus.ir.stmt import SeqStmt, ForStmt, ForThreadGroupStmt, IfStmt, WhileStmt, BreakStmt, InstructionStmt
from tilus.ir.inst import Instruction
from tilus.ir.value import Value, RegisterValue, SharedValue, SharedLayout
from tilus.ir.functors import IRFunctor
from tilus.extensions.hidet.utils.doc import doc_strip_parentheses, doc_join_lines, doc_comment


class IRPrinter(IRFunctor):
    def __init__(self) -> None:
        from hidet.ir.tools import IRPrinter as HidetIRPrinter

        super().__init__()
        self.printer = HidetIRPrinter()
        self.value2name: Dict[Value, str] = {}
        self.comment2key: Dict[str, str] = {}
        self.keys: Set[str] = set()

    def add_key_comment(self, key_hint: str, comment: Any) -> str:
        comment = str(comment)
        if comment in self.comment2key:
            return self.comment2key[comment]
        i = 0
        while True:
            key = key_hint + "_" + str(i)
            if key not in self.keys:
                self.keys.add(key)
                self.comment2key[comment] = key
                return key
            i += 1

    def get_value_type(self, value: Value) -> Doc:
        if isinstance(value, RegisterValue):
            doc = Text("register, ")
            doc += self.printer(value.dtype) + "[" + self.visit(value.shape) + "], "
            doc += "local_size={}".format(value.layout.local_size)
            doc += ", {}".format(self.visit(value.layout))
            return doc
        elif isinstance(value, SharedValue):
            doc = Text("shared, ")
            doc += self.printer(value.dtype) + "[" + self.visit(value.shape) + "], "
            doc += "size={}".format(value.layout.size)
            doc += ", {}".format(self.visit(value.layout))
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
            if isinstance(value, tuple):
                value_doc = "[" + value_doc + "]"
            if isinstance(value, dict):
                value_doc = "{" + value_doc + "}"
            items.append(key_doc + ": " + doc_strip_parentheses(value_doc))
        return doc_join(items, ", ")

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> Doc:
        if isinstance(node, str):
            return Text(repr(node))
        else:
            return Text(str(node))

    def visit_Expr(self, expr: Expr) -> Doc:
        return self.printer(expr)

    def visit_BaseType(self, tp: BaseType) -> Doc:
        return self.printer(tp)

    def visit_Program(self, prog: Program) -> Doc:
        doc = Doc()

        for func in prog.functions.values():
            doc += self.visit(func) + NewLine()

        return doc

    def visit_Function(self, func: Function) -> Doc:
        # head doc
        head_doc = doc_join_lines(
            seq=[self.visit(p) + ": " + self.printer(p.type) for p in func.params],
            left="def " + func.name + "(",
            right=")",
        )

        # attr doc
        num_blocks_doc = Text("num_blocks = ") + self.visit(func.num_blocks)
        num_warps_doc = Text("num_warps = ") + self.visit(func.num_warps)

        # block mapping doc
        # block_mapping_doc = self.visit(func.block_mapping)

        # weight transform doc
        # weight_transform_doc = doc_join_lines(
        #     seq=[
        #         doc_join_lines(
        #             seq=[self.visit(transform) for transform in transforms], left=self.visit(param) + ": [", right="]"
        #         )
        #         for param, transforms in func.weight_transforms.items()
        #         if len(transforms) > 0
        #     ],
        #     left="weight_transforms = {",
        #     right="}",
        # )

        # divisibility doc
        # divisibility: Dict[Var, int] = func.var2divisibility
        # divisibility_doc = doc_join_lines(
        #     seq=[self.visit(var) + ": " + str(divisibility[var]) for var in divisibility],
        #     left="divisibility = {",
        #     right="}",
        # )

        # body doc
        body_doc = self.visit(func.body)

        # comment doc
        comment_doc = doc_comment(
            NewLine()
            + doc_join(seq=[key + ": " + comment for comment, key in self.comment2key.items()], sep=NewLine()),
            "# ",
        )

        # attributes parts
        attributes_doc = doc_comment(
            doc_join(
                [
                    num_blocks_doc,
                    num_warps_doc,
                    # weight_transform_doc,
                    # divisibility_doc
                ],
                NewLine(),
            ),
            comment_string="# ",
        )

        # combine them
        doc = doc_join(
            [head_doc, (NewLine() + attributes_doc).indent(4), (NewLine() + body_doc).indent(4), comment_doc], ""
        )
        return doc

    def visit_InstructionStmt(self, stmt: InstructionStmt) -> Doc:
        return self.visit(stmt.inst)

    def visit_SeqStmt(self, stmt: SeqStmt) -> Doc:
        return doc_join([self.visit(node) for node in stmt.seq], NewLine())

    def visit_ForStmt(self, stmt: ForStmt) -> Doc:
        head_doc = Doc()
        if stmt.unroll_factor:
            if stmt.unroll_factor == -1:
                head_doc += "#pragma unroll"
            else:
                head_doc += "#pragma unroll {}".format(stmt.unroll_factor)
            head_doc += NewLine()
        head_doc += Text("for ") + self.printer(stmt.iter_var) + " in range(" + self.visit(stmt.extent) + "):"
        body_doc = NewLine() + self.visit(stmt.body)
        doc = head_doc + body_doc.indent(4)
        return doc

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> Doc:
        head_doc = (
            Text("for ")
            + self.printer(stmt.iter_var)
            + " in thread_groups(num_groups="
            + self.visit(stmt.num_groups)
            + "):"
        )
        body_doc = NewLine() + self.visit(stmt.body)
        doc = head_doc + body_doc.indent(4)
        return doc

    def visit_IfStmt(self, stmt: IfStmt) -> Doc:
        head_doc = Text("if ") + self.visit(stmt.cond) + ":"
        then_doc = (NewLine() + self.visit(stmt.then_body)).indent(4)
        if stmt.else_body is not None:
            else_doc = NewLine() + Text("else:")
            else_doc += (NewLine() + self.visit(stmt.else_body)).indent(4)
        else:
            else_doc = Doc()

        return head_doc + then_doc + else_doc

    def visit_WhileStmt(self, stmt: WhileStmt) -> Doc:
        head_doc = Text("while ") + self.visit(stmt.cond) + ":"
        body_doc = (NewLine() + self.visit(stmt.body)).indent(4)
        doc = head_doc + body_doc
        return doc

    def visit_BreakStmt(self, stmt: BreakStmt) -> Doc:
        return Text("break")

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
            v_doc = doc_strip_parentheses(self.visit(v))
            if isinstance(v, (list, tuple)):
                v_doc = "[" + v_doc + "]"
            elif isinstance(v, dict):
                v_doc = "{" + v_doc + "}"
            items.append("{}: {}".format(k, v_doc))
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
            doc += "  # " + self.get_value_type(inst.output)
        return doc

    def visit_Value(self, value: Value) -> Doc:
        if value not in self.value2name:
            self.value2name[value] = "%" + str(len(self.value2name))
        return Text(self.value2name[value])

    def visit_Layout(self, layout: Layout) -> Doc:
        return Text(self.add_key_comment("layout", str(layout)))

    def visit_SharedLayout(self, node: SharedLayout) -> Doc:
        printer = IRPrinter()
        items = [
            "shape=[" + printer(node.shape) + "]",
            "axes=[" + printer(node.axes) + "]",
            "offset=" + printer(node.offset),
        ]
        doc = Text("SharedLayout(") + doc_join(items, ", ") + ")"
        return Text(self.add_key_comment("shared_layout", doc))
