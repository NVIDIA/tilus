from typing import Any, Sequence

from tilus.ir.functors import IRVisitor
from tilus.ir.node import IRNode


def collect(node: IRNode, types: Sequence[Any]) -> list[Any]:
    visitor = IRVisitor()
    visitor.visit(node)

    types = tuple(types)
    ret = []

    for node in visitor.memo.keys():
        if isinstance(node, types):
            ret.append(node)
    return ret
