"""
Lightweight IR simplifier that used to clean up the IR after transformations.
"""

from hidet.ir.expr import Expr, Var
from tilus.ir.functors import IRRewriter
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    LetStmt,
    SeqStmt,
    Stmt,
)


class IRSimplifier(IRRewriter):
    def visit_SeqStmt(self, stmt: SeqStmt) -> Stmt:
        seq: list[Stmt] = []

        # flatten all nested SeqStmts
        for sub_stmt in stmt.seq:
            if isinstance(sub_stmt, SeqStmt):
                seq.extend(sub_stmt.seq)
            else:
                seq.append(sub_stmt)

        # simplify each sub-statement
        seq = [self.visit(sub_stmt) for sub_stmt in seq]

        # flatten nested SeqStmts again
        new_seq: list[Stmt] = []
        for sub_stmt in seq:
            if isinstance(sub_stmt, SeqStmt):
                new_seq.extend(sub_stmt.seq)
            else:
                new_seq.append(sub_stmt)

        # check if the sequence has changed
        if len(new_seq) == len(stmt.seq) and all(a is b for a, b in zip(new_seq, stmt.seq)):
            return stmt
        else:
            return SeqStmt.create(new_seq)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        bind_vars: list[Var] = []
        bind_values: list[Expr] = []
        body: Stmt = stmt

        # concat multiple LetStmt
        while isinstance(body, LetStmt):
            bind_vars.extend(body.bind_vars)
            bind_values.extend(body.bind_values)
            body = body.body

        # simplify the bind_values and body
        bind_values = [self.visit(value) for value in bind_values]
        body = self.visit(body)

        # flatten nested LetStmt
        while isinstance(body, LetStmt):
            bind_vars.extend(body.bind_vars)
            bind_values.extend(body.bind_values)
            body = body.body

        # check if the LetStmt has changed
        if (
            len(bind_vars) == len(stmt.bind_vars)
            and all(a is b for a, b in zip(bind_vars, stmt.bind_vars))
            and all(a is b for a, b in zip(bind_values, stmt.bind_values))
            and body is stmt.body
        ):
            return stmt
        else:
            return LetStmt.create(bind_vars, bind_values, body)


def simplify(prog: Program) -> Program:
    simplifier = IRSimplifier()
    return simplifier.visit(prog)
