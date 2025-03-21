from __future__ import annotations as _

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from hidet.ir.expr import Expr, Var
from tilus.ir.node import IRNode
from tilus.ir.stmt import Stmt
from tilus.ir.utils import frozendict


@dataclass(frozen=True)
class Metadata:
    divisibility: frozendict[Var, int]

    @staticmethod
    def create(divisibility: Mapping[Var, int]) -> Metadata:
        return Metadata(frozendict(divisibility))


@dataclass(frozen=True, eq=False)
class Function(IRNode):
    name: str
    params: tuple[Var, ...]
    body: Stmt
    num_warps: int
    num_blocks: tuple[Expr, Expr, Expr]
    annotations: frozendict[str, Any]
    metadata: Metadata

    @staticmethod
    def create(
        name: str,
        params: Sequence[Var],
        body: Stmt,
        num_warps: int,
        num_blocks: Sequence[Expr],
        annotations: Mapping[str, Any],
        metadata: Optional[Metadata] = None,
    ) -> Function:
        assert len(num_blocks) == 3
        if metadata is None:
            metadata = Metadata(frozendict())
        return Function(
            name,
            tuple(params),
            body,
            num_warps,
            (num_blocks[0], num_blocks[1], num_blocks[2]),
            frozendict(annotations),
            metadata,
        )

    def with_body(self, new_body: Stmt) -> Function:
        return dataclasses.replace(self, body=new_body)

    def with_name(self, new_name: str) -> Function:
        return dataclasses.replace(self, name=new_name)

    def with_divisibility(self, divisibility: dict[Var, int]) -> Function:
        metadata = Metadata(frozendict(divisibility))
        return dataclasses.replace(self, metadata=metadata)
