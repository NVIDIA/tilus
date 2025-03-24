from typing import Dict

from hidet.ir.dtypes import int32
from hidet.ir.expr import Constant, Expr
from hidet.transforms.rule_based_simplifier import BoundAnalyzer, BoundInfo, RuleBasedSimplifier
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import (
    CopyAsyncGenericInst,
    LoadGlobalGenericInst,
    StoreGlobalGenericInst,
)
from tilus.ir.stmt import ForStmt, ForThreadGroupStmt, IfStmt, SeqStmt, Stmt
from tilus.transforms.base import Pass
from tilus.utils import same_list


class BoundAwareSimplifyRewriter(IRRewriter):
    def __init__(self) -> None:
        super().__init__()
        self.simplifier: RuleBasedSimplifier = RuleBasedSimplifier()
        self.analyzer: BoundAnalyzer = self.simplifier.analyzer
        self.bound: Dict[Expr, BoundInfo] = self.analyzer.bound

    def visit_Function(self, func: Function) -> Function:
        return super().visit_Function(func)

    def visit_Expr(self, expr: Expr) -> Expr:
        return self.simplifier(expr)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        self.analyzer(stmt.extent)
        bound = self.bound[stmt.extent]
        if bound.value is not None and bound.value in [0, 1]:
            if bound.value == 0:
                return SeqStmt(())
            else:
                self.bound[stmt.iter_var] = BoundInfo(value=0)
                self.memo[stmt.iter_var] = int32.zero
                assert self.simplifier.memo is not None
                self.simplifier.memo[stmt.iter_var] = int32.zero
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
            assert self.simplifier.memo is not None
            self.simplifier.memo[stmt.iter_var] = int32.zero
            return self.visit(stmt.body)
        return super().visit_ForThreadGroupStmt(stmt)

    # instructions

    def visit_CopyAsyncGenericInst(self, inst: CopyAsyncGenericInst) -> Instruction:
        for axis, extent in zip(inst.axes, inst.inputs[0].as_shared_tensor().shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
        return super().visit_Instruction(inst)

    def visit_LoadGlobalGenericInst(self, inst: LoadGlobalGenericInst) -> Instruction:
        for axis, extent in zip(inst.axes, inst.register_output.shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
        return super().visit_Instruction(inst)

    def visit_StoreGlobalGenericInst(self, inst: StoreGlobalGenericInst) -> Instruction:
        for axis, extent in zip(inst.axes, inst.inputs[0].as_register_tensor().shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
        return super().visit_Instruction(inst)


class BoundAwareSimplifyPass(Pass):
    def __init__(self):
        super().__init__()

    def __call__(self, prog: Function) -> Function:
        rewriter = BoundAwareSimplifyRewriter()
        return rewriter(prog)


def bound_aware_simplify_pass() -> Pass:
    return BoundAwareSimplifyPass()
