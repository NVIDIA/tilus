from typing import List, Optional, Sequence

from dataclasses import dataclass
from hidet.ir.expr import Expr, Var
from tilus.ir.inst import Instruction
from tilus.ir.node import IRNode


@dataclass(frozen=True, eq=False)
class Stmt(IRNode):
    pass


@dataclass(frozen=True, eq=False)
class SeqStmt(Stmt):
    seq: tuple[Stmt, ...]

    @staticmethod
    def create(seq: Sequence[Stmt]):
        return SeqStmt(tuple(seq))


@dataclass(frozen=True, eq=False)
class ForStmt(Stmt):
    iter_var: Var
    extent: Expr
    body: Stmt

    # candidates:
    # - None (no annotation),
    # - -1 (unroll all),
    # - n (n >= 1, unroll with factor n)
    unroll_factor: Optional[int]


@dataclass(frozen=True, eq=False)
class ForThreadGroupStmt(Stmt):
    iter_var: Var
    num_groups: int
    body: Stmt


@dataclass(frozen=True, eq=False)
class IfStmt(Stmt):
    cond: Expr
    then_body: Stmt
    else_body: Optional[Stmt]

    def with_else_body(self, else_body: Stmt):
        return IfStmt(self.cond, self.then_body, else_body)


@dataclass(frozen=True, eq=False)
class WhileStmt(Stmt):
    cond: Expr
    body: Stmt


@dataclass(frozen=True, eq=False)
class BreakStmt(Stmt):
    pass


@dataclass(frozen=True, eq=False)
class InstructionStmt(Stmt):
    inst: Instruction


def seq_stmt(seq: Sequence[Stmt | Instruction]) -> Stmt:
    stmt_seq: List[Stmt] = [InstructionStmt(item) if isinstance(item, Instruction) else item for item in seq]
    if len(stmt_seq) == 1:
        return stmt_seq[0]
    else:
        return SeqStmt(tuple(stmt_seq))
