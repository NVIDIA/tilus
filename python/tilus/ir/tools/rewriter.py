from typing import Any, Mapping

from tilus.ir.functors import IRRewriter
from tilus.ir.node import IRNode


class SimpleRewriter(IRRewriter):
    """
    A simple rewriter that applies a function to each instruction in the program.
    """

    def __init__(self, rewrite_map: Mapping[IRNode, Any]):
        super().__init__()
        self.memo.update(rewrite_map)


def rewrite(node: IRNode, rewrite_map: Mapping[Any, Any]) -> Any:
    """
    Rewrite the components of the given node using the provided rewrite map.

    Parameters
    ----------
    node: IRNode
        The node to rewrite.

    rewrite_map: Mapping[IRNode, IRNode]
        A mapping from nodes to their rewritten versions.

    Returns
    -------
    ret: Any
        The rewritten node.
    """
    rewriter = SimpleRewriter(rewrite_map)
    return rewriter.visit(node)
