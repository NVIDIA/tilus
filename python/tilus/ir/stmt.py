from typing import List, Union, Optional

from hidet.ir.expr import Expr, Var
from .inst import Instruction


class Stmt:
    pass


class SeqStmt(Stmt):
    def __init__(self, seq) -> None:
        self.seq: List[Union[Stmt, Instruction]] = seq

        assert all(isinstance(s, (Stmt, Instruction)) for s in seq)


class ForStmt(Stmt):
    def __init__(self, iter_var: Var, extent: Expr, body: Stmt, unroll_factor: Optional[int]):
        self.iter_var: Var = iter_var
        self.extent: Expr = extent
        self.body: Stmt = body

        # candidates:
        # - None (no annotation),
        # - -1 (unroll all),
        # - n (n >= 1, unroll with factor n)
        self.unroll_factor: Optional[int] = unroll_factor

        assert isinstance(iter_var, Var) and isinstance(extent, Expr) and isinstance(body, Stmt)
        assert unroll_factor is None or (
            not isinstance(unroll_factor, bool)
            and isinstance(unroll_factor, int)
            and (unroll_factor == -1 or unroll_factor >= 1)
        ), unroll_factor


class ForThreadGroupStmt(Stmt):
    def __init__(self, iter_var: Var, num_groups: int, body: Stmt):
        self.iter_var: Var = iter_var
        self.num_groups: int = num_groups
        self.body: Stmt = body

        assert isinstance(iter_var, Var) and isinstance(num_groups, int) and isinstance(body, Stmt)


class IfStmt(Stmt):
    def __init__(self, cond: Expr, then_body: Stmt, else_body: Optional[Stmt] = None):
        self.cond: Expr = cond
        self.then_body: Stmt = then_body
        self.else_body: Optional[Stmt] = else_body

        assert (
            isinstance(cond, (Expr, bool))
            and isinstance(then_body, Stmt)
            and (else_body is None or isinstance(else_body, Stmt))
        )


class WhileStmt(Stmt):
    def __init__(self, cond: Expr, body: Stmt):
        self.cond: Expr = cond
        self.body: Stmt = body

        assert isinstance(cond, Expr) and isinstance(body, Stmt)


class BreakStmt(Stmt):
    pass
