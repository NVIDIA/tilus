from typing import Optional, Union, Sequence
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, var, Var
from hidet.ir.stmt import ForStmt, WhileStmt, DeclareStmt, AssertStmt, BreakStmt, ForStmtAttr, Stmt, DeclareScope
from hidet.ir.builders.stmt_builder import StmtBuilder as OriginalStmtBuilder, StmtScope


class StmtBuilder(OriginalStmtBuilder):
    def append(self, stmt: Union[Stmt, Expr, Sequence[Stmt], None]) -> None:
        if stmt is None:
            return
        else:
            super().append(stmt)

    def declare_var(
        self, name: str, tp: BaseType, init: Optional[Expr] = None, scope: Optional[DeclareScope] = None
    ) -> Var:
        v = var(name, tp)
        self.append(DeclareStmt(v, init=init, scope=scope))
        return v

    def assertion(self, cond: Expr, msg: str) -> None:
        self.append(AssertStmt(cond, msg))

    def comment(self, comment_string: str, style: str = "//") -> None:
        from tilus.extensions.hidet.ir.primitives.debug import comment

        self.append(comment(comment_string, style=style))

    def brk(self) -> None:
        self.append(BreakStmt())

    def for_grid(self, shape: Sequence[Union[Expr, int]]) -> StmtScope:
        return super().for_grid(list(shape))

    def for_range(self, extent: Union[Expr, int], *, attr: Optional[str | ForStmtAttr] = None) -> StmtScope:
        iter_var = var("i")
        if isinstance(attr, str):
            attr = ForStmtAttr.parse(attr, num_loops=1)[0]
        return StmtScope(self, stmts=ForStmt(iter_var, extent, attr=attr), ret=iter_var)

    def while_loop(self, cond: Expr) -> StmtScope:
        return StmtScope(self, stmts=WhileStmt(cond, body=None), ret=None)  # type: ignore
