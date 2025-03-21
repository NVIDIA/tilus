import dataclasses
from typing import Any, Dict, Hashable, List, Tuple, TypeVar, Union

from hidet.ir.expr import Expr
from hidet.ir.type import BaseType
from tilus.ir.func import Function
from tilus.ir.inst import (
    AllocateGlobalInst,
    AllocateRegisterInst,
    AllocateSharedInst,
    AssignInst,
    BroadcastElementwiseBinaryInst,
    CastInst,
    CopyAsyncCommitGroupInst,
    CopyAsyncInst,
    CopyAsyncWaitAllInst,
    CopyAsyncWaitGroupInst,
    ElementwiseBinaryInst,
    ElementwiseUnaryInst,
    ExitInst,
    FormatPrintInst,
    FreeSharedInst,
    GlobalViewInst,
    Instruction,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadMatrixInst,
    LoadSharedInst,
    MmaDotInst,
    PrintValueInst,
    ShuffleDownInst,
    ShuffleUpInst,
    SimtDotInst,
    StoreGlobalGenericInst,
    StoreGlobalInst,
    StoreSharedInst,
    SyncReduceThreadsInst,
    SyncThreadsInst,
    ViewInst,
    ViewSharedInst,
)
from tilus.ir.layout import GlobalLayout, RegisterLayout
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    BreakStmt,
    DeclareStmt,
    ForStmt,
    ForThreadGroupStmt,
    IfStmt,
    InstStmt,
    SeqStmt,
    Stmt,
    TensorPtrStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedLayout, SharedTensor
from tilus.utils import same_list

InstClsVar = TypeVar("InstClsVar", bound=Instruction)


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
        elif isinstance(node, InstStmt):
            ret = self.visit_InstStmt(node)
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
        elif isinstance(node, DeclareStmt):
            ret = self.visit_DeclareStmt(node)
        elif isinstance(node, AssignStmt):
            ret = self.visit_AssignStmt(node)
        elif isinstance(node, TensorPtrStmt):
            ret = self.visit_TensorPtrStmt(node)
        # instruction
        elif isinstance(node, Instruction):
            ret = self.visit_Instruction(node)
        # scalar expression and type
        elif isinstance(node, Expr):
            ret = self.visit_Expr(node)
        elif isinstance(node, BaseType):
            ret = self.visit_BaseType(node)
        # value and layout
        elif isinstance(node, RegisterTensor):
            ret = self.visit_RegisterTensor(node)
        elif isinstance(node, SharedTensor):
            ret = self.visit_SharedTensor(node)
        elif isinstance(node, GlobalTensor):
            ret = self.visit_GlobalTensor(node)
        elif isinstance(node, RegisterLayout):
            ret = self.visit_RegisterLayout(node)
        elif isinstance(node, SharedLayout):
            ret = self.visit_SharedLayout(node)
        elif isinstance(node, GlobalLayout):
            ret = self.visit_GlobalLayout(node)
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

    def visit_InstStmt(self, stmt: InstStmt) -> Any:
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

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Any:
        raise NotImplementedError()

    def visit_AssignStmt(self, stmt: AssignStmt) -> Any:
        raise NotImplementedError()

    def visit_TensorPtrStmt(self, stmt: TensorPtrStmt) -> Any:
        raise NotImplementedError()

    # values

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> Any:
        raise NotImplementedError()

    def visit_SharedTensor(self, tensor: SharedTensor) -> Any:
        raise NotImplementedError()

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> Any:
        raise NotImplementedError()

    def visit_RegisterLayout(self, layout: RegisterLayout) -> Any:
        raise NotImplementedError()

    def visit_SharedLayout(self, node: SharedLayout) -> Any:
        raise NotImplementedError()

    def visit_GlobalLayout(self, node: GlobalLayout) -> Any:
        raise NotImplementedError()

    # instructions

    def visit_AssignInst(self, inst: AssignInst) -> Any:
        raise NotImplementedError()

    def visit_AllocateRegisterInst(self, inst: AllocateRegisterInst) -> Any:
        raise NotImplementedError()

    def visit_AllocateGlobalInst(self, inst: AllocateGlobalInst) -> Any:
        raise NotImplementedError()

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst) -> Any:
        raise NotImplementedError()

    def visit_FreeSharedInst(self, inst: FreeSharedInst) -> Any:
        raise NotImplementedError()

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> Any:
        raise NotImplementedError()

    def visit_LoadGlobalGenericInst(self, inst: LoadGlobalGenericInst) -> Any:
        raise NotImplementedError()

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst) -> Any:
        raise NotImplementedError()

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> Any:
        raise NotImplementedError()

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> Any:
        raise NotImplementedError()

    def visit_StoreGlobalGenericInst(self, inst: StoreGlobalGenericInst) -> Any:
        raise NotImplementedError()

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> Any:
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

    def visit_GlobalViewInst(self, inst: GlobalViewInst) -> Any:
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

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst) -> Any:
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
        updated = type(node)({key: self.visit(value) for key, value in node.items()})
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
            ],
            [
                func.body,
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
                metadata=func.metadata,
            )

    def visit_InstStmt(self, stmt: InstStmt) -> Stmt:
        inst_or_stmt = self.visit(stmt.inst)
        if isinstance(inst_or_stmt, Stmt):
            return inst_or_stmt
        elif isinstance(inst_or_stmt, Instruction):
            return InstStmt(inst_or_stmt)
        else:
            raise ValueError(f"An instruction should be rewritten to an instruction or a statement, got {inst_or_stmt}")

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

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Stmt:
        init = self.visit(stmt.init)
        if init is stmt.init:
            return stmt
        else:
            return DeclareStmt(stmt.var, init)

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        value = self.visit(stmt.value)
        if value is stmt.value:
            return stmt
        else:
            return AssignStmt(stmt.var, value)

    def visit_TensorPtrStmt(self, stmt: TensorPtrStmt) -> Stmt:
        tensor = self.visit(stmt.tensor)
        if tensor is stmt.tensor:
            return stmt
        else:
            return TensorPtrStmt(stmt.ptr_var, tensor)

    def visit_WhileStmt(self, stmt: WhileStmt) -> Stmt:
        cond = self.visit(stmt.cond)
        body = self.visit(stmt.body)
        if cond is stmt.cond and body is stmt.body:
            return stmt
        else:
            return WhileStmt(cond, body)

    def default_visit_Instruction(self, inst: InstClsVar) -> InstClsVar:
        output = self.visit(inst.output)
        inputs = self.visit(inst.inputs)
        attributes = self.visit(inst.attributes)

        if output is inst.output and inputs is inst.inputs and attributes is inst.attributes:
            return inst
        else:
            return dataclasses.replace(inst, output=output, inputs=inputs, **attributes)

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> RegisterTensor:
        layout = self.visit(tensor.layout)
        if layout is tensor.layout:
            return tensor
        else:
            return RegisterTensor.create(dtype=tensor.dtype, layout=layout)

    def visit_SharedTensor(self, tensor: SharedTensor) -> SharedTensor:
        layout = self.visit(tensor.layout)
        if layout is tensor.layout:
            return tensor
        else:
            return SharedTensor.create(dtype=tensor.dtype, layout=layout)

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> GlobalTensor:
        layout = self.visit(tensor.layout)
        if layout is tensor.layout:
            return tensor
        else:
            return GlobalTensor.create(dtype=tensor.dtype, layout=layout)

    def visit_RegisterLayout(self, layout: RegisterLayout) -> RegisterLayout:
        return layout

    def visit_SharedLayout(self, layout: SharedLayout) -> SharedLayout:
        return layout

    def visit_GlobalLayout(self, layout: GlobalLayout) -> GlobalLayout:
        shape = self.visit(layout.shape)
        size = self.visit(layout.size)
        offset = self.visit(layout.offset)

        if shape is layout.shape and offset is layout.offset and size is layout.size:
            return layout
        else:
            return GlobalLayout(shape=shape, size=size, axes=layout.axes, offset=offset)

    # instructions

    def visit_AllocateRegisterInst(self, inst: AllocateRegisterInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_AssignInst(self, inst: AssignInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_FreeSharedInst(self, inst: FreeSharedInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalGenericInst(self, inst: LoadGlobalGenericInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalGenericInst(self, inst: StoreGlobalGenericInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_CastInst(self, inst: CastInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseUnaryInst(self, inst: ElementwiseUnaryInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ElementwiseBinaryInst(self, inst: ElementwiseBinaryInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_BroadcastElementwiseBinaryInst(self, inst: BroadcastElementwiseBinaryInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_MmaDotInst(self, inst: MmaDotInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_SimtDotInst(self, inst: SimtDotInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_PrintValueInst(self, inst: PrintValueInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_FormatPrintInst(self, inst: FormatPrintInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ShuffleUpInst(self, inst: ShuffleUpInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ShuffleDownInst(self, inst: ShuffleDownInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ViewInst(self, inst: ViewInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ViewSharedInst(self, inst: ViewSharedInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_GlobalViewInst(self, inst: GlobalViewInst) -> Any:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncCommitGroupInst(self, inst: CopyAsyncCommitGroupInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitGroupInst(self, inst: CopyAsyncWaitGroupInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_CopyAsyncWaitAllInst(self, inst: CopyAsyncWaitAllInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_SyncThreadsInst(self, inst: SyncThreadsInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst) -> Union[Instruction, Stmt]:
        return self.default_visit_Instruction(inst)

    def visit_ExitInst(self, inst: ExitInst) -> Union[Instruction, Stmt]:
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

    def visit_InstStmt(self, stmt: InstStmt) -> None:
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

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.visit(stmt.var)
        self.visit(stmt.init)

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        self.visit(stmt.var)
        self.visit(stmt.value)

    def visit_TensorPtrStmt(self, stmt: TensorPtrStmt) -> None:
        self.visit(stmt.ptr_var)
        self.visit(stmt.tensor)

    # values

    def visit_RegisterTensor(self, tensor: RegisterTensor) -> None:
        pass

    def visit_SharedTensor(self, tensor: SharedTensor) -> None:
        self.visit(tensor.layout)

    def visit_GlobalTensor(self, tensor: GlobalTensor) -> Any:
        self.visit(tensor.layout)
        self.visit(tensor.layout)

    def visit_RegisterLayout(self, layout: RegisterLayout) -> None:
        pass

    def visit_SharedLayout(self, layout: SharedLayout) -> None:
        self.visit(layout.offset)

    def visit_GlobalLayout(self, layout: GlobalLayout) -> None:
        self.visit(layout.shape)
        self.visit(layout.offset)

    # instructions
    def default_visit_Instruction(self, inst: Instruction) -> None:
        self.visit(inst.output)
        self.visit(inst.inputs)
        self.visit(inst.attributes)

    def visit_AllocateRegisterInst(self, inst: AllocateRegisterInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AssignInst(self, inst: AssignInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_AllocateSharedInst(self, inst: AllocateSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_FreeSharedInst(self, inst: FreeSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadGlobalGenericInst(self, inst: LoadGlobalGenericInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadMatrixInst(self, inst: LoadMatrixInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_StoreGlobalGenericInst(self, inst: StoreGlobalGenericInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> None:
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

    def visit_GlobalViewInst(self, inst: GlobalViewInst) -> None:
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

    def visit_SyncReduceThreadsInst(self, inst: SyncReduceThreadsInst) -> None:
        return self.default_visit_Instruction(inst)

    def visit_ExitInst(self, inst: ExitInst) -> None:
        return self.default_visit_Instruction(inst)
