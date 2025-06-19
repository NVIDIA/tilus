from collections import defaultdict

from hidet.ir import (
    Add,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    Cast,
    Constant,
    DeclareStmt,
    Div,
    FloorDiv,
    LogicalNot,
    Multiply,
    SeqStmt,
    Sub,
    TensorElement,
)
from hidet.ir.expr import Address, BinaryExpr, Call, Expr, UnaryExpr, Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import AssignStmt, EvaluateStmt, LetStmt, Stmt
from hidet.transforms.base import FunctionPass
from hidet.utils import repeat_until_converge, same_list


class DeadcodeAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.usage_count: dict[Var, int] = defaultdict(int)
        self.num_declares: dict[Var, int] = defaultdict(int)
        self.num_assigns: dict[Var, int] = defaultdict(int)
        self.num_lets: dict[Var, int] = defaultdict(int)
        self.no_side_effect: dict[Expr, bool] = defaultdict(lambda: False)

    def reset(self):
        self.usage_count.clear()
        self.num_declares.clear()
        self.num_assigns.clear()
        self.num_lets.clear()
        self.no_side_effect.clear()
        self.memo.clear()

    def visit_Var(self, var: Var) -> None:
        self.usage_count[var] += 1
        self.no_side_effect[var] = True
        super().visit_Var(var)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.num_declares[stmt.var] += 1
        self.visit(stmt.init)

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        self.num_assigns[stmt.var] += 1
        self.visit(stmt.value)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.num_lets[bind_var] += 1
            self.visit(bind_value)
        self.visit(stmt.body)

    def visit_Call(self, call: Call) -> None:
        self.no_side_effect[call] = False  # we assume calls have side effects, todo: refine this
        super().visit_Call(call)

    def visit_binary(self, e: BinaryExpr) -> None:
        self.visit(e.a)
        self.visit(e.b)
        self.no_side_effect[e] = self.no_side_effect[e.a] and self.no_side_effect[e.b]

    def visit_unary(self, e: UnaryExpr) -> None:
        self.visit(e.a)
        self.no_side_effect[e] = self.no_side_effect[e.a]

    def visit_Add(self, e: Add) -> None:
        self.visit_binary(e)

    def visit_Sub(self, e: Sub) -> None:
        self.visit_binary(e)

    def visit_Multiply(self, e: Multiply) -> None:
        self.visit_binary(e)

    def visit_Div(self, e: Div) -> None:
        self.visit_binary(e)

    def visit_FloorDiv(self, e: FloorDiv) -> None:
        self.visit_binary(e)

    def visit_BitwiseAnd(self, e: BitwiseAnd) -> None:
        self.visit_binary(e)

    def visit_BitwiseOr(self, e: BitwiseOr) -> None:
        self.visit_binary(e)

    def visit_BitwiseXor(self, e: BinaryExpr) -> None:
        self.visit_binary(e)

    def visit_BitwiseNot(self, e: BitwiseNot) -> None:
        self.visit_unary(e)

    def visit_Not(self, e: LogicalNot) -> None:
        self.visit_unary(e)

    def visit_Address(self, e: Address) -> None:
        self.visit(e.expr)
        self.no_side_effect[e] = self.no_side_effect[e.expr]

    def visit_Constant(self, e: Constant) -> None:
        # Constants are considered to have no side effects
        self.no_side_effect[e] = True
        super().visit_Constant(e)

    def visit_Cast(self, e: Cast) -> None:
        self.visit(e.expr)
        self.no_side_effect[e] = self.no_side_effect[e.expr]

    def visit_TensorElement(self, e: TensorElement) -> None:
        self.visit(e.base)
        for index in e.indices:
            self.visit(index)
        self.no_side_effect[e] = self.no_side_effect[e.base] and all(self.no_side_effect[index] for index in e.indices)


class DeadcodeEliminationRewriter(IRRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.analyzer = DeadcodeAnalyzer()

    def visit_Function(self, func: Function) -> Function:
        self.analyzer.reset()
        self.analyzer.visit(func)
        return super().visit_Function(func)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Stmt:
        if self.analyzer.num_assigns[stmt.var] + self.analyzer.usage_count[stmt.var] == 0:
            if stmt.init is None:
                return SeqStmt([])
            else:
                if self.analyzer.no_side_effect[stmt.init]:
                    return SeqStmt([])
                else:
                    return EvaluateStmt(stmt.init)
        else:
            return super().visit_DeclareStmt(stmt)

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        if self.analyzer.usage_count[stmt.var] == 0:
            if self.analyzer.no_side_effect[stmt.value]:
                return SeqStmt([])
            else:
                return EvaluateStmt(stmt.value)
        else:
            return super().visit_AssignStmt(stmt)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            assert self.analyzer.num_assigns[bind_var] == 0
            assert self.analyzer.num_declares[bind_var] == 0
            if self.analyzer.usage_count[bind_var] == 0 and self.analyzer.no_side_effect[bind_value]:
                continue
            else:
                bind_vars.append(bind_var)
                bind_values.append(bind_value)
        body = self.visit(stmt.body)
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            if len(bind_vars) == 0:
                assert len(bind_values) == 0
                return body
            else:
                return LetStmt(
                    bind_vars=bind_vars,
                    bind_values=bind_values,
                    body=body,
                )

    def visit_EvaluateStmt(self, stmt: EvaluateStmt) -> Stmt:
        if self.analyzer.no_side_effect[stmt.expr]:
            return SeqStmt([])
        else:
            return super().visit_EvaluateStmt(stmt)


class DeadcodeEliminationPass(FunctionPass):
    """
    A pass that eliminates dead code in a function.
    It removes variables that are declared but never used, and statements that have no side effects.
    """

    def process_func(self, func: Function) -> Function:
        rewriter = DeadcodeEliminationRewriter()
        return repeat_until_converge(rewriter, func, limit=5)


def deadcode_elimination_pass() -> FunctionPass:
    """
    Create a dead code elimination pass.

    Returns
    -------
    pass_: FunctionPass
        The dead code elimination pass.
    """
    return DeadcodeEliminationPass()
