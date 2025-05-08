from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from hidet.ir.expr import Constant, Expr
from hidet.ir.tools.simplifier import Simplifier
from hidet.utils.doc import Doc, NewLine
from tilus.ir.functors import IRVisitor
from tilus.ir.instructions import LoadSharedInst, StoreSharedInst
from tilus.ir.node import IRNode
from tilus.ir.prog import Program
from tilus.ir.tools.printer import IRPrinter
from tilus.ir.utils import vector


@dataclass
class Diagnostic:
    kind: str
    context: IRNode
    message: str
    items: Sequence[Any]

    def __init__(self, kind: str, context: IRNode, message: str, items: Sequence[Any]):
        assert kind == "error"
        if len(items) != message.count("{}"):
            raise ValueError("Number of items does not match number of placeholders in message.")
        for item in items:
            assert (
                isinstance(item, IRNode)
                or isinstance(item, (int, float, bool, str, Expr))
                or isinstance(item, (list, tuple, dict))
            ), type(item)
        self.kind = kind
        self.context = context
        self.message = message
        self.items = items


class Diagnostics:
    def __init__(self, prog: Program, diagnostics: Sequence[Diagnostic]):
        self.prog: Program = prog
        self.diagnostics: list[Diagnostic] = list(diagnostics)

    def __bool__(self):
        return bool(self.diagnostics)

    def __str__(self):
        printer = IRPrinter()
        printer(self.prog)
        kind_count: dict[str, int] = defaultdict(int)

        doc = Doc()
        doc += "Verification failed with {} errors:".format(len(self.diagnostics))
        doc += NewLine() + printer(self.prog)
        for diag in self.diagnostics:
            doc += NewLine() + "In " + printer(diag.context) + ":"
            components = diag.message.split("{}")
            doc += NewLine() + "{} {}: ".format(diag.kind.capitalize(), kind_count[diag.kind] + 1)
            kind_count[diag.kind] += 1

            items = list(diag.items)
            for idx, component in enumerate(components):
                doc += component
                if idx == len(components) - 1:
                    continue
                doc += printer(items.pop(0))
            doc += NewLine()

        return str(doc)


class VerificationError(Exception):
    def __init__(self, diagnostics: Diagnostics):
        self.diagnostics = diagnostics

    def __str__(self):
        return str(self.diagnostics)


class IRVerifier(IRVisitor):
    def __init__(self):
        super().__init__()
        self.simplifier: Simplifier = Simplifier()
        self.diagnostics: list[Diagnostic] = []

    def const_or_zero(self, expr: Expr | int) -> int:
        expr = self.simplifier(expr)
        if isinstance(expr, Constant):
            assert isinstance(expr.value, int)
            return int(expr.value)
        return 0

    def error(self, context: IRNode, message: str, *items: Any) -> None:
        self.append("error", context, message, *items)

    def append(self, kind: str, context: IRNode, message: str, *items: Any) -> None:
        self.diagnostics.append(Diagnostic(kind, context, message, items))

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> None:
        shared_shape = vector(inst.shared_input.shape)
        register_shape = vector(inst.register_output.shape)

        if len(shared_shape) != len(register_shape) or any(vector(shared_shape) != vector(register_shape)):
            self.error(
                inst,
                "The shared tensor shape [{}] does not match the register tensor shape [{}].",
                inst.shared_input.shape,
                inst.register_output.shape,
            )

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> None:
        shared_shape = vector(inst.inputs[0].as_shared_tensor().shape)
        register_shape = vector(inst.inputs[1].as_register_tensor().shape)

        if len(shared_shape) != len(register_shape) or any(vector(shared_shape) != vector(register_shape)):
            self.error(
                inst,
                "The shared tensor shape [{}] does not match the register tensor shape [{}].",
                inst.inputs[0].as_shared_tensor().shape,
                inst.inputs[1].as_register_tensor().shape,
            )


def verify(prog: Program) -> None:
    verifier = IRVerifier()
    verifier(prog)
    if verifier.diagnostics:
        raise VerificationError(Diagnostics(prog=prog, diagnostics=verifier.diagnostics))
