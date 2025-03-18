from typing import Any, Sequence, Mapping
import dataclasses
from dataclasses import dataclass
from hidet.ir.expr import Var, Expr
from tilus.ir.stmt import Stmt
from tilus.ir.node import IRNode
from tilus.ir.utils import frozendict


@dataclass(frozen=True, eq=False)
class Function(IRNode):
    name: str
    params: tuple[Var, ...]
    body: Stmt
    num_warps: int
    num_blocks: tuple[Expr, Expr, Expr]
    annotations: frozendict[str, Any]

    @staticmethod
    def create(
        name: str,
        params: Sequence[Var],
        body: Stmt,
        num_warps: int,
        num_blocks: Sequence[Expr],
        annotations: Mapping[str, Any],
    ):
        assert len(num_blocks) == 3
        return Function(
            name, tuple(params), body, num_warps, (num_blocks[0], num_blocks[1], num_blocks[2]), frozendict(annotations)
        )

    def with_body(self, new_body: Stmt):
        return dataclasses.replace(self, body=new_body)
