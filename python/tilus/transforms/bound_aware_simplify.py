from typing import Dict

from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Constant, Expr, LessEqual, LessThan
from hidet.ir.functors import IRRewriter as HidetIRRewriter
from hidet.transforms.rule_based_simplifier import BoundAnalyzer, BoundInfo, RuleBasedSimplifier

from tilus.ir.analyzers import ScalarSet, ScalarSetAnalyzer, analyze_scalar
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import (
    CopyAsyncGenericInst,
    LoadGlobalGenericInst,
    StoreGlobalGenericInst,
)
from tilus.ir.stmt import ForStmt, ForThreadGroupStmt, IfStmt, LetStmt, SeqStmt, Stmt
from tilus.transforms.base import Pass
from tilus.utils import same_list


class ScalarSetBasedSimplifier(HidetIRRewriter):
    def __init__(self, analyzer: ScalarSetAnalyzer):
        super().__init__()
        self.analyzer = analyzer

    def visit_LessEqual(self, e: LessEqual) -> Expr:
        a = self.visit(e.a)
        b = self.visit(e.b)
        sa = self.analyzer(a)
        sb = self.analyzer(b)
        if sa.upper_bound is not None and sb.lower_bound is not None and sa.upper_bound <= sb.lower_bound:
            return boolean.true
        if sa.lower_bound is not None and sb.upper_bound is not None and sa.lower_bound > sb.upper_bound:
            return boolean.false
        if a is e.a and b is e.b:
            return e
        else:
            return LessEqual(a, b)

    def visit_LessThan(self, e: LessThan) -> Expr:
        a = self.visit(e.a)
        b = self.visit(e.b)
        sa = self.analyzer(a)
        sb = self.analyzer(b)
        if sa.upper_bound is not None and sb.lower_bound is not None and sa.upper_bound < sb.lower_bound:
            return boolean.true
        if sa.lower_bound is not None and sb.upper_bound is not None and sa.lower_bound >= sb.upper_bound:
            return boolean.false
        if a is e.a and b is e.b:
            return e
        else:
            return LessThan(a, b)


class BoundAwareSimplifyRewriter(IRRewriter):
    def __init__(self) -> None:
        super().__init__()
        self.bound_info_simplifier: RuleBasedSimplifier = RuleBasedSimplifier()
        self.analyzer: BoundAnalyzer = self.bound_info_simplifier.analyzer
        self.bound: Dict[Expr, BoundInfo] = self.analyzer.bound
        self.scalar_set_analyzer: ScalarSetAnalyzer = ScalarSetAnalyzer({})
        self.scalar_set_simplifier: ScalarSetBasedSimplifier = ScalarSetBasedSimplifier(self.scalar_set_analyzer)

    def visit_Function(self, func: Function) -> Function:
        func = analyze_scalar(func)
        analysis = func.metadata.analysis
        variables = set(analysis.divisibility) | set(analysis.lower_bound) | set(analysis.upper_bound)
        for var in variables:
            # initialize BoundInfo analyzer
            lb = analysis.lower_bound.get(var, None)
            ub = analysis.upper_bound.get(var, None)
            self.bound[var] = BoundInfo(min_value=lb, max_value=ub)

            # initialize ScalarSet analyzer
            self.scalar_set_analyzer.var2info[var] = ScalarSet(
                divisibility=analysis.divisibility.get(var, 1),
                lower_bound=lb,
                upper_bound=ub,
            )

        return super().visit_Function(func)

    def visit_Expr(self, expr: Expr) -> Expr:
        expr = self.scalar_set_simplifier(expr)
        expr = self.bound_info_simplifier(expr)
        return expr

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(bind_value)
            if isinstance(bind_value, Constant):
                self.memo[bind_var] = bind_value
                self.bound_info_simplifier.memo[bind_var] = bind_value
            else:
                bind_vars.append(bind_var)
                bind_values.append(bind_value)
        body = self.visit(stmt.body)

        if body is stmt.body and same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values):
            return stmt
        else:
            if len(bind_vars) == 0:
                return body
            else:
                return LetStmt.create(bind_vars=bind_vars, bind_values=bind_values, body=body)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        self.analyzer(stmt.extent)
        bound = self.bound[stmt.extent]
        if bound.value is not None and bound.value in [0, 1]:
            if bound.value == 0:
                return SeqStmt(())
            else:
                self.bound[stmt.iter_var] = BoundInfo(value=0)
                self.memo[stmt.iter_var] = int32.zero
                assert self.bound_info_simplifier.memo is not None
                self.bound_info_simplifier.memo[stmt.iter_var] = int32.zero
                return self.visit(stmt.body)
        else:
            return super().visit_ForStmt(stmt)

    def visit_IfStmt(self, stmt: IfStmt) -> Stmt:
        cond = self.visit(stmt.cond)
        if isinstance(cond, Constant):
            if cond:
                return self.visit(stmt.then_body)
            else:
                if stmt.else_body is None:
                    return SeqStmt(())
                else:
                    return self.visit(stmt.else_body)
        else:
            return super().visit_IfStmt(stmt)

    def visit_SeqStmt(self, stmt: SeqStmt) -> Stmt:
        seq: list[Stmt] = []
        for s in stmt.seq:
            s = self.visit(s)
            if isinstance(s, SeqStmt) and len(s.seq) == 0:
                continue
            elif isinstance(s, SeqStmt):
                seq.extend(s.seq)
            else:
                seq.append(s)
        if same_list(seq, stmt.seq):
            return stmt
        else:
            return SeqStmt.create(seq)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> Stmt:
        if stmt.num_groups == 1:
            self.bound[stmt.iter_var] = BoundInfo(value=0)
            self.memo[stmt.iter_var] = int32.zero
            assert self.bound_info_simplifier.memo is not None
            self.bound_info_simplifier.memo[stmt.iter_var] = int32.zero
            return self.visit(stmt.body)
        return super().visit_ForThreadGroupStmt(stmt)

    # instructions

    def visit_CopyAsyncGenericInst(self, inst: CopyAsyncGenericInst) -> Instruction:
        for axis, extent in zip(inst.axes, inst.inputs[0].as_shared_tensor().shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
            self.scalar_set_analyzer.var2info[axis] = ScalarSet(divisibility=1, lower_bound=0, upper_bound=extent - 1)
        return super().visit_Instruction(inst)

    def visit_LoadGlobalGenericInst(self, inst: LoadGlobalGenericInst) -> Instruction:
        for axis, extent in zip(inst.axes, inst.register_output.shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
            self.scalar_set_analyzer.var2info[axis] = ScalarSet(divisibility=1, lower_bound=0, upper_bound=extent - 1)
        return super().visit_Instruction(inst)

    def visit_StoreGlobalGenericInst(self, inst: StoreGlobalGenericInst) -> Instruction:
        for axis, extent in zip(inst.axes, inst.inputs[0].as_register_tensor().shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
            self.scalar_set_analyzer.var2info[axis] = ScalarSet(divisibility=1, lower_bound=0, upper_bound=extent - 1)
        return super().visit_Instruction(inst)


class BoundAwareSimplifyPass(Pass):
    def __init__(self):
        super().__init__()

    def __call__(self, prog: Function) -> Function:
        rewriter = BoundAwareSimplifyRewriter()
        return rewriter(prog)


def bound_aware_simplify_pass() -> Pass:
    return BoundAwareSimplifyPass()
