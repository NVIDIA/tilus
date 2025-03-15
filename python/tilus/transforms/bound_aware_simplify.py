from typing import Dict, Optional
from hidet.ir.dtypes import int32
from hidet.ir.expr import Constant, Expr
from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier, BoundAnalyzer, BoundInfo
from tilus.ir.functors import IRRewriter
from tilus.ir.func import Function, BlockMapping
from tilus.ir.inst import (
    CopyAsyncInst,
    LoadGlobalInst,
)
from tilus.ir.inst import StoreGlobalInst
from tilus.ir.stmt import SeqStmt, ForStmt, ForThreadGroupStmt, IfStmt
from tilus.ir.weight_transform import (
    WeightTransform,
    WeightLayoutTransform,
    WeightLayoutTransformGeneric,
    WeightValueTransform,
    IndexSymbolicMapping,
)
from tilus.transforms.base import Pass
from tilus.utils import same_list


class BoundAwareSimplifyRewriter(IRRewriter):
    def __init__(self) -> None:
        super().__init__()
        self.simplifier: RuleBasedSimplifier = RuleBasedSimplifier()
        self.analyzer: BoundAnalyzer = self.simplifier.analyzer
        self.bound: Dict[Expr, BoundInfo] = self.analyzer.bound

    def visit_Program(self, prog: Function):
        # annotate the bound information for parameters
        for param, attrs in prog.param2attrs.items():
            info = BoundInfo(min_value=attrs.lower, max_value=attrs.upper)
            self.bound[param] = info

        # analyze and annotate the bound information for virtual axes
        self.visit(prog.block_mapping)

        return super().visit_Program(prog)

    def visit_Expr(self, expr: Expr):
        return self.simplifier(expr)

    def visit_BlockMapping(self, node: BlockMapping):
        # analyze the annotate bound information for hardware axes
        self.analyzer(node.hardware_num_blocks)
        for axis, num_blocks in zip(node.hardware_axes, node.hardware_num_blocks):
            max_value: Optional[int] = self.bound[num_blocks].possible_max_value()
            if max_value is not None:
                self.bound[axis] = BoundInfo(min_value=0, max_value=max_value - 1)

        # virtual axes are calculated based on hardware axes, based on the bound information of hardware axes,
        # we can infer the bound information of virtual axes
        self.analyzer(list(node.virtual_axes_values.values()))
        for axis, value in node.virtual_axes_values.items():
            max_value = self.bound[value].possible_max_value()
            if max_value is not None:
                self.bound[axis] = BoundInfo(min_value=0, max_value=max_value)

        return super().visit_BlockMapping(node)

    def visit_WeightLayoutTransformGeneric(self, node: WeightLayoutTransformGeneric):
        self.bound[node.mapping.axis] = BoundInfo(min_value=0, max_value=node.size - 1)
        self.bound[node.reverse_mapping.axis] = BoundInfo(min_value=0, max_value=node.size - 1)
        index = self.visit(node.mapping.index)
        reverse_index = self.visit(node.reverse_mapping.index)
        if same_list([index, reverse_index], [node.mapping.index, node.reverse_mapping.index]):
            return node
        else:
            return WeightLayoutTransformGeneric(
                dtype=node.dtype,
                size=node.size,
                mapping=IndexSymbolicMapping(node.mapping.axis, index),
                reverse_mapping=IndexSymbolicMapping(node.reverse_mapping.axis, reverse_index),
            )

    def visit_WeightLayoutTransform(self, node: WeightLayoutTransform):
        return node

    def visit_WeightValueTransform(self, node: WeightValueTransform):
        return node

    def visit_WeightTransform(self, node: WeightTransform):
        if isinstance(node, WeightLayoutTransform):
            return self.visit_WeightLayoutTransform(node)
        elif isinstance(node, WeightLayoutTransformGeneric):
            return self.visit_WeightLayoutTransformGeneric(node)
        elif isinstance(node, WeightValueTransform):
            return self.visit_WeightValueTransform(node)
        else:
            raise NotImplementedError(node.__class__.__name__)

    def visit_ForStmt(self, stmt: ForStmt):
        self.analyzer(stmt.extent)
        bound = self.bound[stmt.extent]
        if bound.value is not None and bound.value in [0, 1]:
            if bound.value == 0:
                return SeqStmt([])
            else:
                self.bound[stmt.iter_var] = BoundInfo(value=0)
                self.memo[stmt.iter_var] = int32.zero
                self.simplifier.memo[stmt.iter_var] = int32.zero
                return self.visit(stmt.body)
        else:
            return super().visit_ForStmt(stmt)

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit(stmt.cond)
        if isinstance(cond, Constant):
            if cond:
                return self.visit(stmt.then_body)
            else:
                if stmt.else_body is None:
                    return SeqStmt([])
                else:
                    return self.visit(stmt.else_body)
        else:
            return super().visit_IfStmt(stmt)

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = []
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
            return SeqStmt(seq)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt):
        if stmt.num_groups == 1:
            self.bound[stmt.iter_var] = BoundInfo(value=0)
            self.memo[stmt.iter_var] = int32.zero
            self.simplifier.memo[stmt.iter_var] = int32.zero
            return self.visit(stmt.body)
        return super().visit_ForThreadGroupStmt(stmt)

    # instructions

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst):
        for axis, extent in zip(inst.axes, inst.inputs[0].shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
        return super().default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst):
        for axis, extent in zip(inst.axes, inst.output.shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
        return super().default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst):
        for axis, extent in zip(inst.axes, inst.inputs[0].shape):
            self.bound[axis] = BoundInfo(min_value=0, max_value=extent - 1)
        return super().default_visit_Instruction(inst)


class BoundAwareSimplifyPass(Pass):
    def __init__(self):
        super().__init__()

    def __call__(self, prog: Function) -> Function:
        rewriter = BoundAwareSimplifyRewriter()
        return rewriter(prog)


def bound_aware_simplify_pass() -> Pass:
    return BoundAwareSimplifyPass()
