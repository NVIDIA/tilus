from typing import Literal, Optional

from hidet.ir.expr import Constant, Expr, Var, as_expr
from hidet.ir.tools import simplify

from tilus.ir.builders import IRBuilder
from tilus.ir.stmt import DeclareStmt, ForStmt, Stmt


class TilusLoopIterable:
    def generate_loop_statement(self, loop_vars: list[Var], body: Stmt) -> Stmt:
        raise NotImplementedError()

    def num_loop_vars(self) -> int:
        raise NotImplementedError()

    def bind_tuple(self) -> bool:
        return False


class RangeLoop(TilusLoopIterable):
    def __init__(
        self, start: Expr | int, stop: Expr | int, step: Expr | int, unroll: Optional[Literal["all"] | int] = None
    ):
        self.start: Expr = simplify(as_expr(start))
        self.stop: Expr = simplify(as_expr(stop))
        self.step: Expr = simplify(as_expr(step))
        self.unroll: Optional[Literal["all"] | int] = unroll

    def generate_loop_statement(self, loop_vars: list[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == self.num_loop_vars()

        # process unroll
        unroll_factor: Optional[int]
        match self.unroll:
            case None:
                unroll_factor = None
            case "all":
                unroll_factor = -1
            case factor:
                unroll_factor = factor

        if isinstance(self.start, Constant) and self.start == 0 and isinstance(self.step, Constant) and self.step == 1:
            # range(stop)
            return ForStmt(
                iter_var=loop_vars[0],
                extent=self.stop,
                body=body,
                unroll_factor=unroll_factor,
            )
        else:
            # range(start, stop, step)
            ib = IRBuilder()
            loop_var = loop_vars[0]
            assert loop_var.hint is not None

            with ib.for_range(
                extent=(self.stop - self.start + (self.step - 1)) // self.step,
                iter_name_hint=loop_var.hint + "_",
                unroll_factor=unroll_factor,
            ) as i:
                ib.append(DeclareStmt(loop_var, init=self.start + i * self.step))
                ib.append(body)
            return ib.flush_stmts()

    def num_loop_vars(self) -> int:
        return 1

    def bind_tuple(self) -> bool:
        return False


def range(
    start: Expr | int,
    stop: Optional[Expr | int] = None,
    step: Optional[Expr | int] = None,
    /,
    *,
    unroll: Optional[Literal["all"] | int] = None,
) -> RangeLoop:
    if stop is None or step is None:
        if step is not None or stop is not None:
            raise ValueError("stop and step must be specified together.")
        start, stop, step = 0, start, 1
    assert start is not None and stop is not None and step is not None
    return RangeLoop(start, stop, step, unroll)
