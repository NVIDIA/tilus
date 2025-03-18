import dataclasses
from typing import List, Tuple, Dict, Union, Hashable
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from tilus.ir.layout import Layout
from tilus.ir.prog import Program
from tilus.ir.func import Function
from tilus.ir.stmt import SeqStmt, ForStmt, ForThreadGroupStmt, IfStmt, WhileStmt, BreakStmt, InstructionStmt
from tilus.ir.value import Value, RegisterValue, SharedValue, SharedLayout
from tilus.ir.inst import (
    Instruction,
    AllocateInst,
    LoadGlobalInst,
    StoreGlobalInst,
    CastInst,
    ElementwiseUnaryInst,
    ElementwiseBinaryInst,
    MmaDotInst,
    PrintValueInst,
    FormatPrintInst,
    ShuffleUpInst,
    ShuffleDownInst,
    ViewInst,
    CopyAsyncInst,
    AllocateSharedInst,
    ViewSharedInst,
    CopyAsyncCommitGroupInst,
    CopyAsyncWaitGroupInst,
    CopyAsyncWaitAllInst,
    SyncThreadsInst,
    AllocateScalarInst,
    LoadMatrixInst,
    LoadSharedInst,
    AssignScalarInst,
    FreeSharedInst,
    BroadcastElementwiseBinaryInst,
    StoreSharedInst,
    AllocateGlobalInst,
    LoadScalarInst,
    SyncReduceThreadsInst,
    StoreScalarInst,
    ExitInst,
    SimtDotInst,
    AssignInst,
    AtomicScalarInst,
)
from tilus.utils import same_list


class IRFunctor:
    def __init__(self):
        self.memo = {}

    def __call__(self, node):
        return self.visit(node)

    def visit(self, node):
        key: Hashable
        if isinstance(node, (list, tuple, dict)):
            key = id(node)
        elif isinstance(node, (str, int, float, bool)):
            key = (type(node), node)
        else:
            key = node
        if key in self.memo:
            return self.memo[key]

        if isinstance(node, Program):
            ret = self.visit_Program(node)
        elif isinstance(node, Function):
            ret = self.visit_Function(node)
        # statements
        elif isinstance(node, InstructionStmt):
            ret = self.visit_InstructionStmt(node)
        elif isinstance(node, SeqStmt):
            ret = self.visit_SeqStmt(node)
        elif isinstance(node, ForStmt):
            ret = self.visit_ForStmt(node)
        elif isinstance(node, ForThreadGroupStmt):
            ret = self.visit_ForThreadGroupStmt(node)
        elif isinstance(node, IfStmt):
            ret = self.visit_IfStmt(node)
        elif isinstance(node, WhileStmt):
            ret = self.visit_WhileStmt(node)
        elif isinstance(node, BreakStmt):
            ret = self.visit_BreakStmt(node)
        # instruction
        elif isinstance(node, Instruction):
            ret = self.visit_Instruction(node)
        # scalar expression and type
        elif isinstance(node, Expr):
            ret = self.visit_Expr(node)
        elif isinstance(node, BaseType):
            ret = self.visit_BaseType(node)
        # value and layout
        elif isinstance(node, Value):
            ret = self.visit_Value(node)
        elif isinstance(node, Layout):
            ret = self.visit_Layout(node)
        elif isinstance(node, SharedLayout):
            ret = self.visit_SharedLayout(node)
        # python native
        elif isinstance(node, list):
            ret = self.visit_list(node)
        elif isinstance(node, tuple):
            ret = self.visit_tuple(node)
        elif isinstance(node, dict):
            ret = self.visit_dict(node)
        elif isinstance(node, (int, float, bool, str, type(None))):
            ret = self.visit_PyConstant(node)
        else:
            raise NotImplementedError(node.__class__.__name__)

        self.memo[key] = ret
        return ret

    def visit_Value(self, value: Value):
        if isinstance(value, RegisterValue):
            return self.visit_RegisterValue(value)
        elif isinstance(value, SharedValue):
            return self.visit_SharedValue(value)
        else:
            raise NotImplementedError(value.__class__.__name__)

    def visit_Instruction(self, inst: Instruction):
        return getattr(self, "visit_{}".format(inst.__class__.__name__))(inst)

    def visit_list(self, lst: List):
        raise NotImplementedError()

    def visit_tuple(self, lst: Tuple):
        raise NotImplementedError()

    def visit_dict(self, node: Dict):
        raise NotImplementedError()

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]):
        raise NotImplementedError()

    def visit_Expr(self, expr: Expr):
        raise NotImplementedError()

    def visit_BaseType(self, tp: BaseType):
        raise NotImplementedError()

    def visit_Program(self, prog: Program):
        raise NotImplementedError()

    def visit_Function(self, func: Function):
        raise NotImplementedError()

    # statements

    def visit_InstructionStmt(self, stmt: InstructionStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        raise NotImplementedError()

    def visit_ForStmt(self, stmt: ForStmt):
        raise NotImplementedError()

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt):
        raise NotImplementedError()

    def visit_IfStmt(self, stmt: IfStmt):
        raise NotImplementedError()

    def visit_WhileStmt(self, stmt: WhileStmt):
        raise NotImplementedError()

    def visit_BreakStmt(self, stmt: BreakStmt):
        raise NotImplementedError()

    # values

    def visit_RegisterValue(self, value: RegisterValue):
        raise NotImplementedError()

    def visit_SharedValue(self, value: SharedValue):
        raise NotImplementedError()

    def visit_Layout(self, layout: Layout):
        raise NotImplementedError()

    def visit_SharedLayout(self, node: SharedLayout):
        raise NotImplementedError()

    # instructions

    def visit_AllocateInst(self, inst: AllocateInst):
        raise NotImplementedError()

    def visit_AssignInst(self, inst: AssignInst):
        raise NotImplementedError()

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst):
        raise NotImplementedError

    def visit_AllocateScalarInst(self, inst: AllocateScalarInst):
        raise NotImplementedError()

    def visit_FreeSharedInst(self, inst: FreeSharedInst):
        raise NotImplementedError()

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst):
        raise NotImplementedError()

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst):
        raise NotImplementedError()

    def visit_LoadSharedInst(self, inst: LoadSharedInst):
        raise NotImplementedError()

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst):
        raise NotImplementedError()

    def visit_StoreSharedInst(self, inst: StoreSharedInst):
        raise NotImplementedError()

    def visit_AssignScalarInst(self, inst: AssignScalarInst):
        raise NotImplementedError()

    def visit_CastInst(self, inst: CastInst):
        raise NotImplementedError()

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst):
        raise NotImplementedError()

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst):
        raise NotImplementedError()

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst):
        raise NotImplementedError()

    def visit_MmaDotInst(self, inst: MmaDotInst):
        raise NotImplementedError()

    def visit_SimtDotInst(self, inst: SimtDotInst):
        raise NotImplementedError()

    def visit_PrintValueInst(self, inst: PrintValueInst):
        raise NotImplementedError()

    def visit_FormatPrintInst(self, inst: FormatPrintInst):
        raise NotImplementedError()

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst):
        raise NotImplementedError()

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst):
        raise NotImplementedError()

    def visit_ViewInst(self, inst: ViewInst):
        raise NotImplementedError()

    def visit_ViewSharedInst(self, inst: ViewSharedInst):
        raise NotImplementedError()

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst):
        raise NotImplementedError()

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst):
        raise NotImplementedError()

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst):
        raise NotImplementedError()

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst):
        raise NotImplementedError()

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst):
        raise NotImplementedError()

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst):
        raise NotImplementedError()

    def visit_LoadScalarInst(self, inst: LoadScalarInst):
        raise NotImplementedError()

    def visit_AtomicScalarInst(self, inst: AtomicScalarInst):
        raise NotImplementedError()

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst):
        raise NotImplementedError()

    def visit_StoreScalarInst(self, inst: StoreScalarInst):
        raise NotImplementedError()

    def visit_ExitInst(self, inst: ExitInst):
        raise NotImplementedError()


class IRRewriter(IRFunctor):
    def visit_list(self, lst: List):
        updated = [self.visit(item) for item in lst]
        if same_list(lst, updated):
            return lst
        else:
            return updated

    def visit_tuple(self, lst: Tuple):
        updated = tuple(self.visit(item) for item in lst)
        if same_list(lst, updated):
            return lst
        else:
            return updated

    def visit_dict(self, node: Dict):
        updated = {key: self.visit(value) for key, value in node.items()}
        if same_list(list(node.values()), list(updated.values())):
            return node
        else:
            return updated

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]):
        return node

    def visit_Expr(self, expr: Expr):
        return expr

    def visit_BaseType(self, tp: BaseType):
        return tp

    def visit_Program(self, prog: Program):
        functions = self.visit(prog.functions)
        if same_list([functions], [prog.functions]):
            return prog
        else:
            return Program(functions=functions)

    def visit_Function(self, func: Function):
        body = self.visit(func.body)
        # block_mapping = self.visit(func.block_mapping)
        # weight_transforms = self.visit(func.weight_transforms)
        if same_list(
            [
                body,
                # block_mapping, weight_transforms
            ],
            [
                func.body,
                # func.block_mapping, func.weight_transforms
            ],
        ):
            return func
        else:
            return Function(
                name=func.name,
                params=func.params,
                num_warps=func.num_warps,
                num_blocks=func.num_blocks,
                body=body,
                annotations=func.annotations,
            )

    def visit_InstructionStmt(self, stmt: InstructionStmt):
        inst = self.visit(stmt.inst)
        assert isinstance(inst, Instruction)
        return InstructionStmt(inst)

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = self.visit(stmt.seq)
        if seq is stmt.seq:
            return stmt
        else:
            return SeqStmt(seq)

    def visit_ForStmt(self, stmt: ForStmt):
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        if extent is stmt.extent and body is stmt.body:
            return stmt
        else:
            return ForStmt(stmt.iter_var, extent, body, stmt.unroll_factor)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt):
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        else:
            return ForThreadGroupStmt(stmt.iter_var, stmt.num_groups, body)

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body)
        if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        return stmt

    def visit_WhileStmt(self, stmt: WhileStmt):
        cond = self.visit(stmt.cond)
        body = self.visit(stmt.body)
        if cond is stmt.cond and body is stmt.body:
            return stmt
        else:
            return WhileStmt(cond, body)

    def default_visit_Instruction(self, inst: Instruction):
        output = self.visit(inst.output)
        inputs = self.visit(inst.inputs)
        attributes = self.visit(inst.attributes)

        if output is inst.output and inputs is inst.inputs and attributes is inst.attributes:
            return inst
        else:
            return dataclasses.replace(inst, output=output, inputs=inputs, **attributes)

    def visit_Value(self, value: Value):
        return value

    def visit_Layout(self, layout: Layout):
        return layout

    def visit_SharedLayout(self, node: SharedLayout):
        return node

    # instructions

    def visit_AllocateInst(self, inst: AllocateInst):
        return self.default_visit_Instruction(inst)

    def visit_AssignInst(self, inst: AssignInst):
        return self.default_visit_Instruction(inst)

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_AllocateScalarInst(self, inst: AllocateScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_FreeSharedInst(self, inst: FreeSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadSharedInst(self, inst: LoadSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst):
        return self.default_visit_Instruction(inst)

    def visit_StoreSharedInst(self, inst: StoreSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_AssignScalarInst(self, inst: AssignScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_CastInst(self, inst: CastInst):
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst):
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst):
        return self.default_visit_Instruction(inst)

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst):
        return self.default_visit_Instruction(inst)

    def visit_MmaDotInst(self, inst: MmaDotInst):
        return self.default_visit_Instruction(inst)

    def visit_SimtDotInst(self, inst: SimtDotInst):
        return self.default_visit_Instruction(inst)

    def visit_PrintValueInst(self, inst: PrintValueInst):
        return self.default_visit_Instruction(inst)

    def visit_FormatPrintInst(self, inst: FormatPrintInst):
        return self.default_visit_Instruction(inst)

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst):
        return self.default_visit_Instruction(inst)

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst):
        return self.default_visit_Instruction(inst)

    def visit_ViewInst(self, inst: ViewInst):
        return self.default_visit_Instruction(inst)

    def visit_ViewSharedInst(self, inst: ViewSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst):
        return self.default_visit_Instruction(inst)

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst):
        return self.default_visit_Instruction(inst)

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadScalarInst(self, inst: LoadScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_AtomicScalarInst(self, inst: AtomicScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst):
        return self.default_visit_Instruction(inst)

    def visit_StoreScalarInst(self, inst: StoreScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_ExitInst(self, inst: ExitInst):
        return self.default_visit_Instruction(inst)


class IRVisitor(IRFunctor):
    def visit_list(self, lst: List):
        for item in lst:
            self.visit(item)

    def visit_tuple(self, lst: Tuple):
        for item in lst:
            self.visit(item)

    def visit_dict(self, node: Dict):
        for k, v in node.items():
            self.visit(v)

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]):
        pass

    def visit_Expr(self, expr: Expr):
        pass

    def visit_BaseType(self, tp: BaseType):
        pass

    def visit_Program(self, prog: Program):
        self.visit(prog.functions)

    def visit_Function(self, func: Function):
        self.visit(func.body)

    def visit_InstructionStmt(self, stmt: InstructionStmt):
        self.visit(stmt.inst)

    def visit_SeqStmt(self, stmt: SeqStmt):
        for sub_stmt in stmt.seq:
            self.visit(sub_stmt)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit(stmt.extent)
        self.visit(stmt.body)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt):
        self.visit(stmt.body)

    def visit_IfStmt(self, stmt: IfStmt):
        self.visit(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body is not None:
            self.visit(stmt.else_body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        pass

    def visit_WhileStmt(self, stmt: WhileStmt):
        self.visit(stmt.cond)
        self.visit(stmt.body)

    # values

    def visit_RegisterValue(self, value: RegisterValue):
        pass

    def visit_SharedValue(self, value: SharedValue):
        self.visit(value.layout.offset)

    def visit_Layout(self, layout: Layout):
        pass

    def visit_SharedLayout(self, node: SharedLayout):
        self.visit(node.offset)

    # instructions
    def default_visit_Instruction(self, inst: Instruction):
        self.visit(inst.output)
        self.visit(inst.inputs)
        self.visit(inst.attributes)

    def visit_AllocateInst(self, inst: AllocateInst):
        return self.default_visit_Instruction(inst)

    def visit_AssignInst(self, inst: AssignInst):
        return self.default_visit_Instruction(inst)

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_AllocateScalarInst(self, inst: AllocateScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_FreeSharedInst(self, inst: FreeSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadSharedInst(self, inst: LoadSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst):
        return self.default_visit_Instruction(inst)

    def visit_StoreSharedInst(self, inst: StoreSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_AssignScalarInst(self, inst: AssignScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_CastInst(self, inst: CastInst):
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst):
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst):
        return self.default_visit_Instruction(inst)

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst):
        return self.default_visit_Instruction(inst)

    def visit_MmaDotInst(self, inst: MmaDotInst):
        return self.default_visit_Instruction(inst)

    def visit_SimtDotInst(self, inst: SimtDotInst):
        return self.default_visit_Instruction(inst)

    def visit_PrintValueInst(self, inst: PrintValueInst):
        return self.default_visit_Instruction(inst)

    def visit_FormatPrintInst(self, inst: FormatPrintInst):
        return self.default_visit_Instruction(inst)

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst):
        return self.default_visit_Instruction(inst)

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst):
        return self.default_visit_Instruction(inst)

    def visit_ViewInst(self, inst: ViewInst):
        return self.default_visit_Instruction(inst)

    def visit_ViewSharedInst(self, inst: ViewSharedInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst):
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst):
        return self.default_visit_Instruction(inst)

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst):
        return self.default_visit_Instruction(inst)

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst):
        return self.default_visit_Instruction(inst)

    def visit_LoadScalarInst(self, inst: LoadScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_AtomicScalarInst(self, inst: AtomicScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst):
        return self.default_visit_Instruction(inst)

    def visit_StoreScalarInst(self, inst: StoreScalarInst):
        return self.default_visit_Instruction(inst)

    def visit_ExitInst(self, inst: ExitInst):
        return self.default_visit_Instruction(inst)
