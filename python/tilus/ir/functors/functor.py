import dataclasses
from typing import List, Tuple, Dict, Union, Hashable, Any
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from tilus.ir.layout import Layout
from tilus.ir.prog import Program
from tilus.ir.func import Function
from tilus.ir.stmt import SeqStmt, ForStmt, ForThreadGroupStmt, IfStmt, WhileStmt, BreakStmt, InstructionStmt, Stmt
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

    def visit_Value(self, value: Value) -> Any:
        if isinstance(value, RegisterValue):
            return self.visit_RegisterValue(value)
        elif isinstance(value, SharedValue):
            return self.visit_SharedValue(value)
        else:
            raise NotImplementedError(value.__class__.__name__)

    def visit_Instruction(self, inst: Instruction) -> Any:
        return getattr(self, "visit_{}".format(inst.__class__.__name__))(inst)

    def visit_list(self, lst: List) -> Any:
        raise NotImplementedError()

    def visit_tuple(self, lst: Tuple) -> Any:
        raise NotImplementedError()

    def visit_dict(self, node: Dict) -> Any:
        raise NotImplementedError()

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> Any:
        raise NotImplementedError()

    def visit_Expr(self, expr: Expr) -> Any:
        raise NotImplementedError()

    def visit_BaseType(self, tp: BaseType) -> Any:
        raise NotImplementedError()

    def visit_Program(self, prog: Program) -> Any:
        raise NotImplementedError()

    def visit_Function(self, func: Function) -> Any:
        raise NotImplementedError()

    # statements

    def visit_InstructionStmt(self, stmt: InstructionStmt) -> Any:
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt) -> Any:
        raise NotImplementedError()

    def visit_ForStmt(self, stmt: ForStmt) -> Any:
        raise NotImplementedError()

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> Any:
        raise NotImplementedError()

    def visit_IfStmt(self, stmt: IfStmt) -> Any:
        raise NotImplementedError()

    def visit_WhileStmt(self, stmt: WhileStmt) -> Any:
        raise NotImplementedError()

    def visit_BreakStmt(self, stmt: BreakStmt) -> Any:
        raise NotImplementedError()

    # values

    def visit_RegisterValue(self, value: RegisterValue) -> Any:
        raise NotImplementedError()

    def visit_SharedValue(self, value: SharedValue) -> Any:
        raise NotImplementedError()

    def visit_Layout(self, layout: Layout) -> Any:
        raise NotImplementedError()

    def visit_SharedLayout(self, node: SharedLayout) -> Any:
        raise NotImplementedError()

    # instructions

    def visit_AllocateInst(self, inst: AllocateInst) -> Any:
        raise NotImplementedError()

    def visit_AssignInst(self, inst: AssignInst) -> Any:
        raise NotImplementedError()

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst) -> Any:
        raise NotImplementedError

    def visit_AllocateScalarInst(self, inst: AllocateScalarInst) -> Any:
        raise NotImplementedError()

    def visit_FreeSharedInst(self, inst: FreeSharedInst) -> Any:
        raise NotImplementedError()

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> Any:
        raise NotImplementedError()

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst) -> Any:
        raise NotImplementedError()

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> Any:
        raise NotImplementedError()

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> Any:
        raise NotImplementedError()

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> Any:
        raise NotImplementedError()

    def visit_AssignScalarInst(self, inst: AssignScalarInst) -> Any:
        raise NotImplementedError()

    def visit_CastInst(self, inst: CastInst) -> Any:
        raise NotImplementedError()

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst) -> Any:
        raise NotImplementedError()

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst) -> Any:
        raise NotImplementedError()

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst) -> Any:
        raise NotImplementedError()

    def visit_MmaDotInst(self, inst: MmaDotInst) -> Any:
        raise NotImplementedError()

    def visit_SimtDotInst(self, inst: SimtDotInst) -> Any:
        raise NotImplementedError()

    def visit_PrintValueInst(self, inst: PrintValueInst) -> Any:
        raise NotImplementedError()

    def visit_FormatPrintInst(self, inst: FormatPrintInst) -> Any:
        raise NotImplementedError()

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst) -> Any:
        raise NotImplementedError()

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst) -> Any:
        raise NotImplementedError()

    def visit_ViewInst(self, inst: ViewInst) -> Any:
        raise NotImplementedError()

    def visit_ViewSharedInst(self, inst: ViewSharedInst) -> Any:
        raise NotImplementedError()

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst) -> Any:
        raise NotImplementedError()

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst) -> Any:
        raise NotImplementedError()

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst) -> Any:
        raise NotImplementedError()

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst) -> Any:
        raise NotImplementedError()

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst) -> Any:
        raise NotImplementedError()

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst) -> Any:
        raise NotImplementedError()

    def visit_LoadScalarInst(self, inst: LoadScalarInst) -> Any:
        raise NotImplementedError()

    def visit_AtomicScalarInst(self, inst: AtomicScalarInst) -> Any:
        raise NotImplementedError()

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst) -> Any:
        raise NotImplementedError()

    def visit_StoreScalarInst(self, inst: StoreScalarInst) -> Any:
        raise NotImplementedError()

    def visit_ExitInst(self, inst: ExitInst) -> Any:
        raise NotImplementedError()


class IRRewriter(IRFunctor):
    def visit_list(self, lst: List) -> List:
        updated = [self.visit(item) for item in lst]
        if same_list(lst, updated):
            return lst
        else:
            return updated

    def visit_tuple(self, lst: Tuple) -> Tuple:
        updated = tuple(self.visit(item) for item in lst)
        if same_list(lst, updated):
            return lst
        else:
            return updated

    def visit_dict(self, node: Dict) -> Dict:
        updated = {key: self.visit(value) for key, value in node.items()}
        if same_list(list(node.values()), list(updated.values())):
            return node
        else:
            return updated

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> Union[int, float, bool, str, None]:
        return node

    def visit_Expr(self, expr: Expr) -> Expr:
        return expr

    def visit_BaseType(self, tp: BaseType) -> BaseType:
        return tp

    def visit_Program(self, prog: Program) -> Program:
        functions = self.visit(prog.functions)
        if same_list([functions], [prog.functions]):
            return prog
        else:
            return Program(functions=functions)

    def visit_Function(self, func: Function) -> Function:
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

    def visit_InstructionStmt(self, stmt: InstructionStmt) -> Stmt:
        inst = self.visit(stmt.inst)
        assert isinstance(inst, Instruction)
        return InstructionStmt(inst)

    def visit_SeqStmt(self, stmt: SeqStmt) -> Stmt:
        seq = self.visit(stmt.seq)
        if seq is stmt.seq:
            return stmt
        else:
            return SeqStmt(seq)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        if extent is stmt.extent and body is stmt.body:
            return stmt
        else:
            return ForStmt(stmt.iter_var, extent, body, stmt.unroll_factor)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> Stmt:
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        else:
            return ForThreadGroupStmt(stmt.iter_var, stmt.num_groups, body)

    def visit_IfStmt(self, stmt: IfStmt) -> Stmt:
        cond = self.visit(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body)
        if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_BreakStmt(self, stmt: BreakStmt) -> Stmt:
        return stmt

    def visit_WhileStmt(self, stmt: WhileStmt) -> Stmt:
        cond = self.visit(stmt.cond)
        body = self.visit(stmt.body)
        if cond is stmt.cond and body is stmt.body:
            return stmt
        else:
            return WhileStmt(cond, body)

    def default_visit_Instruction(self, inst: Instruction) -> Instruction:
        output = self.visit(inst.output)
        inputs = self.visit(inst.inputs)
        attributes = self.visit(inst.attributes)

        if output is inst.output and inputs is inst.inputs and attributes is inst.attributes:
            return inst
        else:
            return dataclasses.replace(inst, output=output, inputs=inputs, **attributes)

    def visit_Value(self, value: Value) -> Value:
        return value

    def visit_Layout(self, layout: Layout) -> Layout:
        return layout

    def visit_SharedLayout(self, node: SharedLayout) -> SharedLayout:
        return node

    # instructions

    def visit_AllocateInst(self, inst: AllocateInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_AssignInst(self, inst: AssignInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_AllocateScalarInst(self, inst: AllocateScalarInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_FreeSharedInst(self, inst: FreeSharedInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_AssignScalarInst(self, inst: AssignScalarInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_CastInst(self, inst: CastInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_MmaDotInst(self, inst: MmaDotInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_SimtDotInst(self, inst: SimtDotInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_PrintValueInst(self, inst: PrintValueInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_FormatPrintInst(self, inst: FormatPrintInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ViewInst(self, inst: ViewInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ViewSharedInst(self, inst: ViewSharedInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_LoadScalarInst(self, inst: LoadScalarInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_AtomicScalarInst(self, inst: AtomicScalarInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_StoreScalarInst(self, inst: StoreScalarInst) -> Instruction:
        return self.default_visit_Instruction(inst)

    def visit_ExitInst(self, inst: ExitInst) -> Instruction:
        return self.default_visit_Instruction(inst)


class IRVisitor(IRFunctor):
    def visit_list(self, lst: List) -> None:
        for item in lst:
            self.visit(item)

    def visit_tuple(self, lst: Tuple) -> None:
        for item in lst:
            self.visit(item)

    def visit_dict(self, node: Dict) -> None:
        for k, v in node.items():
            self.visit(v)

    def visit_PyConstant(self, node: Union[int, float, bool, str, None]) -> None:
        pass

    def visit_Expr(self, expr: Expr) -> None:
        pass

    def visit_BaseType(self, tp: BaseType) -> None:
        pass

    def visit_Program(self, prog: Program) -> None:
        self.visit(prog.functions)

    def visit_Function(self, func: Function) -> None:
        self.visit(func.body)

    def visit_InstructionStmt(self, stmt: InstructionStmt) -> None:
        self.visit(stmt.inst)

    def visit_SeqStmt(self, stmt: SeqStmt) -> None:
        for sub_stmt in stmt.seq:
            self.visit(sub_stmt)

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        self.visit(stmt.extent)
        self.visit(stmt.body)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> None:
        self.visit(stmt.body)

    def visit_IfStmt(self, stmt: IfStmt) -> None:
        self.visit(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body is not None:
            self.visit(stmt.else_body)

    def visit_BreakStmt(self, stmt: BreakStmt) -> None:
        pass

    def visit_WhileStmt(self, stmt: WhileStmt) -> None:
        self.visit(stmt.cond)
        self.visit(stmt.body)

    # values

    def visit_RegisterValue(self, value: RegisterValue) -> None:
        pass

    def visit_SharedValue(self, value: SharedValue) -> None:
        self.visit(value.layout.offset)

    def visit_Layout(self, layout: Layout) -> None:
        pass

    def visit_SharedLayout(self, node: SharedLayout) -> None:
        self.visit(node.offset)

    # instructions
    def default_visit_Instruction(self, inst: Instruction) -> None:
        self.visit(inst.output)
        self.visit(inst.inputs)
        self.visit(inst.attributes)

    def visit_AllocateInst(self, inst: AllocateInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AssignInst(self, inst: AssignInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AllocateScalarInst(self, inst: AllocateScalarInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_FreeSharedInst(self, inst: FreeSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AssignScalarInst(self, inst: AssignScalarInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_CastInst(self, inst: CastInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_MmaDotInst(self, inst: MmaDotInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_SimtDotInst(self, inst: SimtDotInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_PrintValueInst(self, inst: PrintValueInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_FormatPrintInst(self, inst: FormatPrintInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ViewInst(self, inst: ViewInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ViewSharedInst(self, inst: ViewSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadScalarInst(self, inst: LoadScalarInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AtomicScalarInst(self, inst: AtomicScalarInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_StoreScalarInst(self, inst: StoreScalarInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ExitInst(self, inst: ExitInst) -> None:
        return self.default_visit_Instruction(inst)
