from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Expr, Var
from hidet.ir.type import BaseType, DataType
from tilus.extensions.hidet.ir.expr import as_expr
from tilus.ir.inst import (
    AllocateRegisterInst,
    AllocateSharedInst,
    AssignInst,
    CastInst,
    CopyAsyncCommitGroupInst,
    CopyAsyncInst,
    CopyAsyncWaitAllInst,
    CopyAsyncWaitGroupInst,
    ElementwiseBinaryInst,
    ExitInst,
    FormatPrintInst,
    FreeSharedInst,
    Instruction,
    LoadGlobalGenericInst,
    LoadMatrixInst,
    LoadSharedInst,
    MmaDotInst,
    PrintValueInst,
    StoreGlobalGenericInst,
    StoreSharedInst,
    SyncReduceThreadsInst,
    SyncThreadsInst,
    ViewInst,
    ViewSharedInst,
)
from tilus.ir.layout import RegisterLayout
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
from tilus.ir.tensor import RegisterTensor, SharedLayout, SharedTensor, Tensor


class StmtContext:
    def __init__(self, vb: StmtBuilderCore) -> None:
        self.vb: StmtBuilderCore = vb

    def enter(self) -> None:
        self.vb._stack.append([])

    def pop(self) -> Stmt:
        return SeqStmt(tuple(self.vb._stack.pop()))

    def append(self, stmt: Stmt) -> None:
        self.vb._stack[-1].append(stmt)

    @property
    def innermost_stack(self) -> List[Stmt]:
        return self.vb._stack[-1]


class ForContext(StmtContext):
    def __init__(
        self,
        vb: StmtBuilderCore,
        iter_vars: List[Var],
        extents: List[Expr],
        unrolls: List[Optional[int]],
        unwrap: bool = False,
    ):
        super().__init__(vb)
        self.iter_vars: List[Var] = iter_vars
        self.extents: List[Expr] = extents
        self.unrolls: List[Optional[int]] = unrolls
        self.unwrap: bool = unwrap

    def __enter__(self):
        self.enter()
        if self.unwrap:
            assert len(self.iter_vars) == 1
            return self.iter_vars[0]
        else:
            return self.iter_vars

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return

        body = self.pop()

        for iter_var, extent, unroll in reversed(list(zip(self.iter_vars, self.extents, self.unrolls))):
            body = ForStmt(iter_var, extent, body, unroll)

        self.append(body)


class ForThreadGroupContext(StmtContext):
    def __init__(self, vb: StmtBuilderCore, iter_var: Var, num_groups: int):
        super().__init__(vb)
        self.iter_var: Var = iter_var
        self.num_groups: int = num_groups

    def __enter__(self) -> Var:
        self.enter()
        return self.iter_var

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(ForThreadGroupStmt(self.iter_var, self.num_groups, body=self.pop()))


class IfContext(StmtContext):
    def __init__(self, vb: StmtBuilderCore, cond: Expr):
        super().__init__(vb)
        self.cond: Expr = cond

    def __enter__(self) -> None:
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(IfStmt(self.cond, then_body=self.pop(), else_body=None))


class ElseIfContext(StmtContext):
    def __init__(self, vb: StmtBuilderCore, cond: Expr):
        super().__init__(vb)
        self.cond: Expr = cond

    def __enter__(self) -> None:
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        body = self.pop()
        if_stmt = self.innermost_stack.pop()
        assert isinstance(if_stmt, IfStmt), "with vb.else_if() must be used after with vb.if_then() or vb.else_if()"
        if_chain: List[IfStmt] = [if_stmt]
        while if_chain[-1].else_body is not None:
            else_body = if_chain[-1].else_body
            assert isinstance(else_body, IfStmt), (
                "with vb.else_if() must be used after with vb.if_then() or vb.else_if()"
            )
            if_chain.append(else_body)
        if_chain.append(IfStmt(self.cond, then_body=body, else_body=None))

        # merge the if-chain
        while len(if_chain) > 1:
            # chain, a, b
            b = if_chain.pop()
            a = if_chain.pop()
            if_chain.append(a.with_else_body(b))

        self.append(if_chain[0])


class OtherwiseContext(StmtContext):
    def __init__(self, vb: StmtBuilderCore):
        super().__init__(vb)

    def __enter__(self) -> None:
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        else_body = self.pop()
        if_stmt = self.innermost_stack[-1]
        assert isinstance(if_stmt, IfStmt), "with vb.otherwise() must be used after with vb.if_then() or vb.else_if()"
        if_chain: List[IfStmt] = [if_stmt]
        while if_chain[-1].else_body is not None:
            else_body = if_chain[-1].else_body
            assert isinstance(else_body, IfStmt), (
                "with vb.otherwise() must be used after with vb.if_then() or vb.else_if()"
            )
            if_chain.append(else_body)
        if_chain[-1] = if_chain[-1].with_else_body(else_body)

        # merge the if-chain
        while len(if_chain) > 1:
            # chain, a, b
            b = if_chain.pop()
            a = if_chain.pop()
            if_chain.append(a.with_else_body(b))

        self.append(if_chain[0])


class WhileContext(StmtContext):
    def __init__(self, vb: StmtBuilderCore, cond: Expr):
        super().__init__(vb)
        self.cond: Expr = cond

    def __enter__(self) -> None:
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(WhileStmt(self.cond, body=self.pop()))


class StmtBuilderCore:
    def __init__(self) -> None:
        # context stack
        self._stack: List[List[Stmt]] = [[]]

    def for_range(
        self, extent: Union[Expr, int], iter_name_hint: str = "i", unroll_factor: Optional[int] = None
    ) -> ForContext:
        iter_var = Var(iter_name_hint, type=int32)
        return ForContext(self, [iter_var], [as_expr(extent)], [unroll_factor], unwrap=True)

    def for_grid(self, extents: List[Union[Expr, int]], iter_name_hints: Optional[List[str]] = None) -> ForContext:
        expr_extents = [as_expr(extent) for extent in extents]
        if iter_name_hints is None:
            names = "ijkpqrstuvw"
            assert len(extents) < len(names)
            iter_name_hints = [names[i] for i in range(len(extents))]
        iter_vars = [Var(name, type=int32) for name in iter_name_hints]
        return ForContext(self, iter_vars, expr_extents, unrolls=[None] * len(extents))

    def for_thread_group(self, num_groups: int) -> ForThreadGroupContext:
        iter_var = Var("tg", type=int32)
        return ForThreadGroupContext(self, iter_var, num_groups)

    def while_loop(self, cond: Union[Expr, bool]) -> WhileContext:
        return WhileContext(self, as_expr(cond))

    def if_then(self, cond: Union[Expr, bool]) -> IfContext:
        return IfContext(self, as_expr(cond))

    def else_if(self, cond: Union[Expr, bool]) -> ElseIfContext:
        return ElseIfContext(self, as_expr(cond))

    def otherwise(self) -> OtherwiseContext:
        return OtherwiseContext(self)

    def brk(self):
        stmt = BreakStmt()
        self._stack[-1].append(stmt)

    def declare(self, type: BaseType, init: Optional[Expr | float | int] = None) -> Var:
        var = Var("v", type=type)
        self.append(DeclareStmt(var, as_expr(init) if init is not None else None))
        return var

    def assign(self, var: Var, value: Expr) -> None:
        self.append(AssignStmt(var, value))

    def tensor_ptr(self, tensor: Tensor) -> Var:
        ptr_var = Var("v", type=~tensor.dtype)
        self.append(TensorPtrStmt(ptr_var, tensor))
        return ptr_var

    def append(self, inst_or_stmt: Union[Instruction, Stmt]) -> None:
        if isinstance(inst_or_stmt, Instruction):
            stmt: Stmt = InstStmt(inst_or_stmt)
        else:
            stmt = inst_or_stmt
        self._stack[-1].append(stmt)

    def flush_stmts(self) -> Stmt:
        if len(self._stack) != 1:
            raise ValueError("Unbalanced context stack")
        ret: Stmt
        if len(self._stack[0]) != 1:
            ret = SeqStmt(tuple(self._stack.pop()))
        else:
            stmt_or_inst = self._stack.pop()[0]
            if isinstance(stmt_or_inst, Stmt):
                ret = stmt_or_inst
            else:
                ret = SeqStmt((stmt_or_inst,))
        self._stack = [[]]
        return ret


class StmtBuilder(StmtBuilderCore):
    # register value operations
    def allocate_register(
        self,
        dtype: DataType,
        layout: RegisterLayout,
        f_init: Optional[Callable[[Sequence[Var]], Union[Expr, float, int]]] = None,
    ) -> RegisterTensor:
        wrapped_init: Optional[Callable[[Sequence[Var]], Expr]] = None
        if f_init is not None:

            def wrapped_init(axes: Sequence[Var]) -> Expr:
                return as_expr(f_init(axes))

        inst = AllocateRegisterInst.create(dtype, layout, wrapped_init)
        self.append(inst)
        return inst.register_output

    def assign_register(self, output: RegisterTensor, x: RegisterTensor) -> None:
        inst = AssignInst.create(output, x)
        self.append(inst)

    def view(
        self,
        x: RegisterTensor,
        *,
        layout: Optional[RegisterLayout] = None,
        dtype: Optional[DataType] = None,
        local_offset: Union[Expr, int] = 0,
    ) -> RegisterTensor:
        inst = ViewInst.create(x, layout=layout, dtype=dtype, local_offset=local_offset)
        self.append(inst)
        return inst.register_output

    def cast(
        self,
        x: RegisterTensor,
        *,
        dtype: DataType,
        interleave_width: Optional[int] = None,
        interleave_stride: Optional[int] = None,
        ignore_int4b_xor: bool = False,
    ) -> RegisterTensor:
        if x.dtype == dtype:
            return x
        inst = CastInst.create(
            dtype=dtype,
            x=x,
            interleave_width=interleave_width,
            interleave_stride=interleave_stride,
            ignore_int4b_xor=ignore_int4b_xor,
        )
        self.append(inst)
        return inst.register_output

    def view_shared(
        self, x: SharedTensor, *, indices: List[Expr], layout: SharedLayout, dtype: Optional[DataType] = None
    ) -> SharedTensor:
        inst = ViewSharedInst.create(x, indices, layout, dtype)
        self.append(inst)
        return inst.shared_output

    def copy_async(
        self,
        *,
        dst: SharedTensor,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]],
        evict: Optional[str] = None,
    ) -> None:
        inst = CopyAsyncInst.create(dst, ptr, f_offset, f_mask, evict=evict)
        self.append(inst)

    def copy_async_wait_all(self):
        inst = CopyAsyncWaitAllInst.create()
        self.append(inst)

    def copy_async_commit_group(self):
        inst = CopyAsyncCommitGroupInst.create()
        self.append(inst)

    def copy_async_wait_group(self, n: Union[Expr, int]) -> None:
        inst = CopyAsyncWaitGroupInst.create(as_expr(n))
        self.append(inst)

    def elementwise_binary(
        self, x: RegisterTensor, y: RegisterTensor, op: str, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        inst = ElementwiseBinaryInst.create(x, y, op, output=out)
        self.append(inst)
        return inst.register_output

    def add(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, "+", out=out)

    def sub(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, "-", out=out)

    def mul(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, "*", out=out)

    def div(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, "/", out=out)

    def mod(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, "%", out=out)

    def print_tensor(
        self, msg: str, value: RegisterTensor | SharedTensor, fmt: Optional[str] = None, cond: Expr = boolean.true
    ) -> None:
        inst = PrintValueInst.create(value, cond=cond, msg=msg, fmt=fmt)
        self.append(inst)

    def format_print(
        self, fstring: str, expressions: Sequence[Expr | int | float | str], cond: Optional[Expr] = None
    ) -> None:
        if cond is None:
            cond = boolean.true
        inst = FormatPrintInst.create(cond=cond, fstring=fstring, expressions_=expressions)
        self.append(inst)

    def printf(self, fstring: str, *expressions: Expr | int | float | str, cond: Optional[Expr] = None) -> None:
        self.format_print(fstring=fstring, expressions=expressions, cond=cond)

    def mma_dot(
        self,
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        mma_inst: str,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = MmaDotInst.create(
            a=a, b=b, c=c, mma_inst=mma_inst, warp_spatial=warp_spatial, warp_repeat=warp_repeat, output=output
        )
        self.append(inst)
        return inst.register_output

    # shared value operations

    def allocate_shared(
        self, dtype: DataType, shared_layout: SharedLayout, f_init: Optional[Callable[[List[Var]], Expr]] = None
    ) -> SharedTensor:
        inst = AllocateSharedInst.create(dtype=dtype, shared_layout=shared_layout, f_init=f_init)
        self.append(inst)
        return inst.shared_output

    def free_shared(self, shared_value: SharedTensor) -> None:
        inst = FreeSharedInst.create(shared_value)
        self.append(inst)

    def store_shared(self, dst: SharedTensor, src: RegisterTensor, offsets: Optional[List[Expr]] = None) -> None:
        inst = StoreSharedInst.create(dst, src, offsets)
        self.append(inst)

    def load_shared(
        self,
        src: SharedTensor,
        register_layout: RegisterLayout,
        offsets: Optional[List[Expr]] = None,
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if offsets is None:
            offsets = [int32.zero for _ in range(len(src.shape))]
        inst = LoadSharedInst.create(src=src, register_layout=register_layout, offsets=offsets, output=output)
        self.append(inst)
        return inst.register_output

    def load_matrix(
        self,
        src: SharedTensor,
        *,
        register_layout: RegisterLayout,
        offsets: List[Expr],
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = LoadMatrixInst.create(src=src, register_layout=register_layout, offsets=offsets, output=output)
        self.append(inst)
        return inst.register_output

    # global memory operations
    def store_global_generic(
        self,
        x: RegisterTensor,
        *,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
    ) -> None:
        inst = StoreGlobalGenericInst.create(x=x, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        self.append(inst)

    def load_global_generic(
        self,
        *,
        dtype: DataType,
        layout: RegisterLayout,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = LoadGlobalGenericInst.create(
            dtype=dtype, layout=layout, ptr=ptr, f_offset=f_offset, f_mask=f_mask, out=out
        )
        self.append(inst)
        return inst.register_output

    # control operations

    def syncthreads(self) -> None:
        inst = SyncThreadsInst.create()
        self.append(inst)

    def syncthreads_and(self, cond: Union[Expr, bool]) -> Var:
        inst = SyncReduceThreadsInst.create(reduce_op="and", var_hint="sync_and", reduce_value=as_expr(cond))
        self.append(inst)
        return inst.var

    def syncthreads_or(self, cond: Union[Expr, bool]) -> Var:
        inst = SyncReduceThreadsInst.create(reduce_op="or", var_hint="sync_and", reduce_value=as_expr(cond))
        self.append(inst)
        return inst.var

    def exit(self) -> None:
        inst = ExitInst.create()
        self.append(inst)
