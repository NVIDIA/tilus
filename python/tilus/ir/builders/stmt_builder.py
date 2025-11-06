# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Type, Union

from hidet.ir import primitives
from hidet.ir.dtypes import boolean, int32, uint32
from hidet.ir.expr import BitwiseXor, Equal, Expr, LogicalNot, LessEqual, LessThan, NotEqual, Var, as_expr
from hidet.ir.tools import infer_type
from hidet.ir.type import BaseType, DataType
from hidet.ir.utils import broadcast_shapes, can_broadcast
from hidet.utils import prod, same_list

from tilus.ir.inst import Instruction, InstructionError
from tilus.ir.instructions.annotation import AnnotateLayoutInst
from tilus.ir.instructions.cuda.clc import (
    ClusterLaunchControlGetFirstCtaInst,
    ClusterLaunchControlIsCanceledInst,
    ClusterLaunchControlTryCancelInst,
)
from tilus.ir.instructions.cuda.cluster_sync import ClusterSyncThreadsInst
from tilus.ir.instructions.cuda.cp_async import (
    CopyAsyncCommitGroupInst,
    CopyAsyncGenericInst,
    CopyAsyncInst,
    CopyAsyncWaitAllInst,
    CopyAsyncWaitGroupInst,
)
from tilus.ir.instructions.cuda.cp_async_bulk import (
    CopyAsyncBulkGlobalToClusterSharedInst,
    CopyAsyncBulkGlobalToSharedInst,
    CopyAsyncBulkSharedToClusterSharedInst,
    CopyAsyncBulkSharedToGlobalInst,
)
from tilus.ir.instructions.cuda.cp_async_tensor import (
    CopyAsyncTensorCommitGroupInst,
    CopyAsyncTensorGlobalToSharedInst,
    CopyAsyncTensorSharedToGlobalInst,
    CopyAsyncTensorWaitGroupInst,
)
from tilus.ir.instructions.cuda.ldmatrix import LoadMatrixConfig, LoadMatrixInst
from tilus.ir.instructions.cuda.mbarrier import (
    AllocBarrierInst,
    ArriveBarrierInst,
    ArriveRemoteBarrierInst,
    FenceProxyCopyAsync,
    WaitBarrierInst,
)
from tilus.ir.instructions.cuda.mma_dot import DotInst
from tilus.ir.instructions.cuda.semaphore import LockSemaphoreInst, ReleaseSemaphoreInst
from tilus.ir.instructions.cuda.tcgen05 import (
    Tcgen05AllocInst,
    Tcgen05CommitInst,
    Tcgen05CopyInst,
    Tcgen05DeallocInst,
    Tcgen05LoadInst,
    Tcgen05MmaSSInst,
    Tcgen05MmaTSInst,
    Tcgen05RelinquishAllocPermitInst,
    Tcgen05SliceInst,
    Tcgen05StoreInst,
    Tcgen05ViewInst,
    Tcgen05WaitInst,
)
from tilus.ir.instructions.generic import (
    AddInst,
    AllocateGlobalInst,
    AllocateRegisterInst,
    AllocateSharedInst,
    AssignInst,
    CastInst,
    DivInst,
    ElementwiseBinaryInst,
    ElementwiseUnaryInst,
    ExitInst,
    FormatPrintInst,
    FreeSharedInst,
    GlobalViewInst,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedGenericInst,
    LoadSharedInst,
    ModInst,
    MulInst,
    PermuteSharedInst,
    PrintTensorInst,
    ReduceInst,
    RepeatInst,
    RepeatInterleaveInst,
    SliceAssignInst,
    SliceGlobalInst,
    SliceRegisterInst,
    SliceSharedInst,
    SqueezeInst,
    StoreGlobalGenericInst,
    StoreGlobalInst,
    StoreSharedGenericInst,
    StoreSharedInst,
    SubInst,
    SyncReduceThreadsInst,
    SyncThreadsInst,
    TransposeInst,
    UnsqueezeInst,
    ViewInst,
    WhereInst,
)
from tilus.ir.layout import GlobalLayout, RegisterLayout, global_row_major
from tilus.ir.stmt import (
    AssignStmt,
    BreakStmt,
    DeclareStmt,
    ForStmt,
    IfStmt,
    InstStmt,
    SeqStmt,
    Stmt,
    TensorItemPtrStmt,
    TensorItemValueStmt,
    ThreadGroupStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedLayout, SharedTensor, Tensor, TMemoryTensor


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


class ThreadGroupContext(StmtContext):
    def __init__(self, vb: StmtBuilderCore, group_index: int, group_size: int):
        super().__init__(vb)
        self.group_index: int = group_index
        self.group_size: int = group_size

    def __enter__(self) -> None:
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(ThreadGroupStmt(group_index=self.group_index, group_size=self.group_size, body=self.pop()))


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

    def thread_group(self, group_index: int, group_size: int) -> ThreadGroupContext:
        return ThreadGroupContext(self, group_index=group_index, group_size=group_size)

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

    def declare(self, type: BaseType, init: Optional[Expr | float | int] = None, hint: Optional[str] = None) -> Var:
        if hint is not None:
            hint = "v"
        var = Var(hint, type=type)
        self.append(DeclareStmt(var, as_expr(init) if init is not None else None))
        return var

    def assign(self, var: Var, value: Expr) -> None:
        self.append(AssignStmt(var, value))

    def tensor_item_ptr(self, tensor: Tensor, space: str = "generic") -> Var:
        if space in ["generic", "global"]:
            ptr_var = Var("ptr", type=~tensor.dtype)
        else:
            ptr_var = Var("ptr", int32)
        if isinstance(tensor, (SharedTensor, GlobalTensor)):
            if prod(tensor.shape) != 1:
                raise ValueError("tensor_item_ptr requires tensor with a single element")
        else:
            raise ValueError("tensor_item_ptr only supports SharedTensor and GlobalTensor")
        self.append(TensorItemPtrStmt(ptr_var, tensor, space=space))
        return ptr_var

    def tensor_item_value(self, tensor: Tensor) -> Var:
        if isinstance(tensor, (SharedTensor, GlobalTensor, RegisterTensor)):
            if prod(tensor.shape) != 1:
                raise ValueError("tensor_item_value requires tensor with a single element")
        else:
            raise ValueError("tensor_item_value only supports SharedTensor, GlobalTensor and RegisterTensor")
        var = Var("val", type=tensor.dtype)
        self.append(TensorItemValueStmt(var, tensor))
        return var

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
    def tensor_ptr(self, tensor: Tensor, space: str = "generic") -> Var:
        if isinstance(tensor, GlobalTensor):
            tensor = self.slice_global(
                tensor, offsets=[0 for _ in range(len(tensor.shape))], slice_dims=[], slice_shape=[]
            )
        elif isinstance(tensor, SharedTensor):
            tensor = self.slice_shared(
                tensor, offsets=[0 for _ in range(len(tensor.shape))], slice_dims=[], slice_shape=[]
            )
        else:
            raise ValueError("tensor_ptr only supports GlobalTensor and SharedTensor")
        return self.tensor_item_ptr(tensor, space=space)

    # register value operations
    def allocate_register(
        self,
        dtype: DataType,
        *,
        shape: Optional[Sequence[int]] = None,
        layout: Optional[RegisterLayout] = None,
        f_init: Optional[Callable[[Sequence[Var]], Union[Expr, float, int]]] = None,
    ) -> RegisterTensor:
        wrapped_f_init: Optional[Callable[[Sequence[Var]], Expr]] = None
        if f_init is not None:

            def wrapped_f_init_(axes: Sequence[Var]) -> Expr:
                return as_expr(f_init(axes))

            wrapped_f_init = wrapped_f_init_

        if shape is None:
            if layout is None:
                raise ValueError("Either shape or layout must be provided")
            shape = layout.shape

        output = RegisterTensor.create(dtype, shape=shape, optional_layout=layout)

        inst = AllocateRegisterInst.create(output=output, f_init=wrapped_f_init)
        self.append(inst)
        return inst.register_output

    def allocate_global(
        self,
        dtype: DataType,
        shape: Sequence[int | Expr],
        *,
        layout: Optional[GlobalLayout] = None,
        requires_clean: bool,
    ) -> GlobalTensor:
        if layout is None:
            assert shape is not None
            layout = global_row_major(*shape)
        inst = AllocateGlobalInst.create(
            output=GlobalTensor.create(dtype=dtype, layout=layout),
            require_clean=requires_clean,
        )
        self.append(inst)
        return inst.global_output

    def slice_global(
        self,
        tensor: GlobalTensor,
        offsets: Sequence[Expr | int],
        slice_dims: Sequence[int],
        slice_shape: Sequence[Expr | int],
    ) -> GlobalTensor:
        offsets_ = [as_expr(offset) for offset in offsets]
        inst = SliceGlobalInst.create(
            tensor=tensor,
            offsets=offsets_,
            dims=slice_dims,
            shape=slice_shape,
        )
        self.append(inst)
        return inst.global_output

    def assign_register(self, output: RegisterTensor, x: RegisterTensor) -> None:
        inst = AssignInst.create(output, x)
        self.append(inst)

    def slice_assign_register(
        self,
        output: RegisterTensor,
        x: RegisterTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        inst = SliceAssignInst.create(
            dst=output,
            src=x,
            offsets=[as_expr(offset) for offset in offsets],
            dims=dims,
        )
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

    def squeeze(
        self,
        x: RegisterTensor,
        dim: int | Sequence[int],
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = SqueezeInst.create(x=x, dims=dim, out=out)
        self.append(inst)
        return inst.register_output

    def unsqueeze(
        self,
        x: RegisterTensor,
        dim: int | Sequence[int],
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = UnsqueezeInst.create(x=x, dims=dim, out=out)
        self.append(inst)
        return inst.register_output

    def cast(
        self,
        x: RegisterTensor,
        *,
        dtype: DataType,
    ) -> RegisterTensor:
        if x.dtype == dtype:
            return x
        inst = CastInst.create(
            x=x,
            output=RegisterTensor.create(dtype=dtype, shape=x.shape, optional_layout=x.optional_layout),
        )
        self.append(inst)
        return inst.register_output

    def copy_async(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> None:
        inst = CopyAsyncInst.create(
            src=src, dst=dst, offsets=offsets, dims=dims, evict=evict, check_bounds=check_bounds
        )
        self.append(inst)

    def copy_async_generic(
        self,
        *,
        dst: SharedTensor,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]],
        evict: Optional[str] = None,
    ) -> None:
        inst = CopyAsyncGenericInst.create(dst, ptr, f_offset, f_mask, evict=evict)
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

    def copy_async_bulk_global_to_shared(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]],
        mbarrier: Expr,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> None:
        if dims is None:
            dims = list(range(len(src.shape)))
        inst = CopyAsyncBulkGlobalToSharedInst.create(
            src=src, dst=dst, offsets=offsets, dims=dims, mbarrier=mbarrier, evict=evict, check_bounds=check_bounds
        )
        self.append(inst)

    def copy_async_bulk_global_to_cluster_shared(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        mbarrier: Expr,
        cta_mask: int,
        dims: Optional[Sequence[int]],
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> None:
        if dims is None:
            if len(src.shape) != len(dst.shape):
                raise InstructionError(
                    "When `dims` is not specified, the rank of src and dst must be the same, "
                    f"but got {len(src.shape)} and {len(dst.shape)}"
                )
            dims = list(range(len(src.shape)))
        inst = CopyAsyncBulkGlobalToClusterSharedInst.create(
            src=src,
            dst=dst,
            offsets=offsets,
            dims=dims,
            mbarrier=mbarrier,
            cta_mask=cta_mask,
            evict=evict,
            check_bounds=check_bounds,
        )
        self.append(inst)

    def copy_async_bulk_shared_to_cluster_shared(
        self,
        src: SharedTensor,
        dst: SharedTensor,
        remote_rank: int,
        mbarrier: Expr,
    ) -> None:
        inst = CopyAsyncBulkSharedToClusterSharedInst.create(
            src=src,
            dst=dst,
            remote_rank=remote_rank,
            mbarrier=mbarrier,
        )
        self.append(inst)

    def copy_async_bulk_shared_to_global(
        self,
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]],
        check_bounds: bool = True,
    ) -> None:
        if dims is None:
            if len(src.shape) != len(dst.shape):
                raise InstructionError(
                    "When `dims` is not specified, the rank of src and dst must be the same, "
                    f"but got {len(src.shape)} and {len(dst.shape)}"
                )
            dims = list(range(len(src.shape)))
        inst = CopyAsyncBulkSharedToGlobalInst.create(
            src=src, dst=dst, offsets=offsets, dims=dims, check_bounds=check_bounds
        )
        self.append(inst)

    def copy_async_tensor_global_to_shared(
        self,
        *,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        mbarrier: Expr | RegisterTensor,
        cache_policy: Optional[Expr] = None,
    ) -> None:
        if dims is None:
            dims = list(range(len(src.shape)))
        if isinstance(mbarrier, RegisterTensor):
            mbarrier = self.tensor_item_value(mbarrier)
        inst = CopyAsyncTensorGlobalToSharedInst.create(
            src=src, dst=dst, offsets=offsets, dims=dims, mbarrier=mbarrier, cache_policy=cache_policy
        )
        self.append(inst)

    def copy_async_tensor_shared_to_global(
        self,
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        cache_policy: Optional[Expr] = None,
    ) -> None:
        if dims is None:
            dims = list(range(len(src.shape)))
        inst = CopyAsyncTensorSharedToGlobalInst.create(
            src=src, dst=dst, offsets=offsets, dims=dims, cache_policy=cache_policy
        )
        self.append(inst)

    def copy_async_tensor_commit_group(self):
        inst = CopyAsyncTensorCommitGroupInst.create()
        self.append(inst)

    def copy_async_tensor_wait_group(self, n: int) -> None:
        inst = CopyAsyncTensorWaitGroupInst.create(n)
        self.append(inst)

    def elementwise_binary(
        self,
        x: RegisterTensor,
        y: RegisterTensor,
        *,
        f_compute: Callable[[Var, Var], Expr],
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        from hidet.ir.utils.broadcast_utils import broadcast_shapes, can_mutually_broadcast

        if not can_mutually_broadcast(x.shape, y.shape):
            raise InstructionError(f"Cannot broadcast {x.shape} and {y.shape}")
        if out is None:
            lhs = Var("x", x.dtype)
            rhs = Var("y", y.dtype)
            value = f_compute(lhs, rhs)
            out_shape = [int(s) for s in broadcast_shapes([x.shape, y.shape])]
            out = RegisterTensor.create(dtype=infer_type(value), shape=out_shape)

        inst = ElementwiseBinaryInst.create(x, y, f_compute=f_compute, output=out)
        self.append(inst)
        return inst.register_output

    def elementwise_unary(
        self, x: RegisterTensor, *, f_compute: Callable[[Var], Expr], out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        if out is None:
            arg: Var = Var("x", x.dtype)
            value: Expr = f_compute(arg)
            out = RegisterTensor.create(dtype=infer_type(value), shape=x.shape)

        inst = ElementwiseUnaryInst.create(x=x, f_compute=f_compute, output=out)
        self.append(inst)
        return inst.register_output

    def repeat(
        self,
        x: RegisterTensor,
        repeats: Sequence[int],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if out is None:
            shape: Sequence[int] = x.shape
            if len(repeats) > len(shape):
                shape = [1] * (len(repeats) - len(shape)) + list(shape)
            if len(repeats) < len(shape):
                repeats = [1] * (len(shape) - len(repeats)) + list(repeats)
            shape = [a * b for a, b in zip(shape, repeats)]
            out = RegisterTensor.create(dtype=x.dtype, shape=shape)
        inst = RepeatInst.create(x=x, output=out)
        self.append(inst)
        return inst.register_output

    def repeat_interleave(
        self,
        x: RegisterTensor,
        repeats: Sequence[int],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        from tilus.ir.layout.ops.register_ops import local, unsqueeze

        if out is None:
            layout = x.layout
            if len(repeats) > len(layout.shape):
                layout = unsqueeze(layout, dims=list(range(len(repeats) - len(layout.shape))))
            if len(repeats) < len(layout.shape):
                repeats = [1] * (len(layout.shape) - len(repeats)) + list(repeats)
            layout = layout * local(*repeats)
            out = RegisterTensor.create(dtype=x.dtype, shape=layout.shape, optional_layout=layout)
        inst = RepeatInterleaveInst.create(x=x, output=out)
        self.append(inst)
        return inst.register_output

    def transpose(self, x: RegisterTensor) -> RegisterTensor:
        inst = TransposeInst.create(x)
        self.append(inst)
        return inst.register_output

    def slice_register(
        self,
        tensor: RegisterTensor,
        offsets: Sequence[Expr | int],
        slice_dims: Sequence[int],
        slice_shape: Sequence[int],
    ) -> RegisterTensor:
        offsets_ = [as_expr(offset) for offset in offsets]
        inst = SliceRegisterInst.create(
            tensor=tensor,
            offsets=offsets_,
            dims=slice_dims,
            shape=slice_shape,
        )
        self.append(inst)
        return inst.register_output

    def reduce(
        self,
        x: RegisterTensor,
        *,
        dim: int,
        keepdim: bool,
        op: str,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if out is None:
            shape = list(x.shape)
            if keepdim:
                shape[dim] = 1
            else:
                shape.pop(dim)
            if op in ("min", "max", "sum"):
                dtype = x.dtype
            elif op in ("any", "all"):
                dtype = boolean
            else:
                raise NotImplementedError(f"Unsupported reduction operation: {op}")
            out = RegisterTensor.create(dtype=dtype, shape=shape)
        inst = ReduceInst.create(x=x, output=out, dim=dim, op=op, keepdim=keepdim)
        self.append(inst)
        return inst.register_output

    def _binary(
        self,
        x: RegisterTensor,
        y: RegisterTensor,
        inst_cls: Type[AddInst | SubInst | MulInst | DivInst | ModInst],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        from hidet.ir.utils.broadcast_utils import broadcast_shape, can_mutually_broadcast

        if not can_mutually_broadcast(x.shape, y.shape):
            raise InstructionError(f"Cannot broadcast shape {x.shape} and {y.shape} (op={inst_cls.__name__})")
        if out is None:
            lhs = Var("x", x.dtype)
            rhs = Var("y", y.dtype)
            # used x as output to create inst_cls(), it is not used
            value = inst_cls.create(x, y, x).f_compute(lhs, rhs)
            out_shape = [int(s) for s in broadcast_shape(x.shape, y.shape)]
            out = RegisterTensor.create(dtype=infer_type(value), shape=out_shape)
        inst = inst_cls.create(x=x, y=y, output=out)
        self.append(inst)
        return inst.register_output

    def add(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._binary(x, y, AddInst, out=out)

    def sub(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._binary(x, y, SubInst, out=out)

    def mul(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._binary(x, y, MulInst, out=out)

    def div(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._binary(x, y, DivInst, out=out)

    def mod(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._binary(x, y, ModInst, out=out)

    def less_than(
        self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: LessThan(a, b), out=out)

    def less_equal(
        self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: LessEqual(a, b), out=out)

    def greater_than(
        self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: LessThan(b, a), out=out)

    def greater_equal(
        self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: LessEqual(b, a), out=out)

    def equal(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: Equal(a, b), out=out)

    def not_equal(
        self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: NotEqual(a, b), out=out)

    def bitwise_xor(
        self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: BitwiseXor(a, b), out=out)

    def maximum(self, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_binary(x, y, f_compute=lambda a, b: primitives.max(a, b), out=out)

    def where(
        self, cond: RegisterTensor, x: RegisterTensor, y: RegisterTensor, *, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        if out is None:
            out_shape = [int(s) for s in broadcast_shapes([cond.shape, x.shape, y.shape])]
            out = RegisterTensor.create(dtype=y.dtype, shape=out_shape)
        for operand in [x, y, cond]:
            if not can_broadcast(operand.shape, out.shape):
                raise InstructionError(f"Cannot broadcast {operand.shape} to {out.shape}")
        inst = WhereInst.create(cond, x, y, output=out)
        self.append(inst)
        return inst.register_output

    def clip(
        self,
        x: RegisterTensor,
        min: Expr | int | float,
        max: Expr | int | float,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        min = x.dtype(min)
        max = x.dtype(max)
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.max(primitives.min(arg, max), min), out=out)

    def round(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.round(arg), out=out)

    def sqrt(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.sqrt(arg), out=out)

    def neg(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: -arg, out=out)

    def abs(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.abs(arg), out=out)

    def exp(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.exp(arg), out=out)

    def exp2(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.math.exp2(arg), out=out)

    def log(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: primitives.log(arg), out=out)

    def logical_not(self, x: RegisterTensor, *, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self.elementwise_unary(x, f_compute=lambda arg: LogicalNot(arg), out=out)

    def print_tensor(self, msg: str, tensor: Tensor, fmt: Optional[str] = None, cond: Expr = boolean.true) -> None:
        inst = PrintTensorInst.create(tensor, cond=cond, msg=msg, fmt=fmt)
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

    def dot(
        self,
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        # config: MmaDotConfig,
        # warp_spatial: Sequence[int],
        # warp_repeat: Sequence[int],
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if output is None:
            output = RegisterTensor.create(dtype=c.dtype, shape=c.shape)
        inst = DotInst.create(
            a=a,
            b=b,
            c=c,
            output=output,
        )
        self.append(inst)
        return inst.register_output

    # shared value operations

    def allocate_shared(
        self, dtype: DataType, shape: Sequence[int], layout: Optional[SharedLayout] = None
    ) -> SharedTensor:
        inst = AllocateSharedInst.create(output=SharedTensor.create(dtype=dtype, shape=shape, optional_layout=layout))
        self.append(inst)
        return inst.shared_output

    def free_shared(self, shared_value: SharedTensor) -> None:
        inst = FreeSharedInst.create(shared_value)
        self.append(inst)

    def permute_shared(self, tensor: SharedTensor, dims: Sequence[int]) -> SharedTensor:
        inst = PermuteSharedInst.create(tensor, dims)
        self.append(inst)
        return inst.shared_output

    def slice_shared(
        self,
        tensor: SharedTensor,
        offsets: Sequence[Expr | int],
        slice_dims: Sequence[int],
        slice_shape: Sequence[int],
    ) -> SharedTensor:
        offsets_ = [as_expr(offset) for offset in offsets]
        inst = SliceSharedInst.create(
            tensor=tensor,
            offsets=offsets_,
            dims=slice_dims,
            shape=slice_shape,
        )
        self.append(inst)
        return inst.shared_output

    def load_shared(
        self,
        src: SharedTensor,
        layout: Optional[RegisterLayout] = None,
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if output is None:
            output = RegisterTensor.create(dtype=src.dtype, shape=src.shape, optional_layout=layout)
        inst = LoadSharedInst.create(x=src, output=output)
        self.append(inst)
        return inst.register_output

    def store_shared(
        self,
        dst: SharedTensor,
        src: RegisterTensor,
    ) -> None:
        inst = StoreSharedInst.create(dst=dst, src=src)
        self.append(inst)

    def load_matrix(
        self,
        smem_addr: Var,
        axes: Sequence[Var],
        offset: Expr,
        config: LoadMatrixConfig,
        output: RegisterTensor,
    ) -> RegisterTensor:
        inst = LoadMatrixInst.create(
            smem_addr=smem_addr,
            axes=axes,
            offset=offset,
            config=config,
            output=output,
        )
        self.append(inst)
        return inst.register_output

    # global memory operations
    def global_view(self, ptr: Expr, dtype: DataType, layout: GlobalLayout) -> GlobalTensor:
        inst = GlobalViewInst.create(output=GlobalTensor.create(dtype=dtype, layout=layout), ptr=ptr)
        self.append(inst)
        return inst.global_output

    def load_global(
        self,
        x: GlobalTensor,
        *,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        shape: Sequence[int],
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if output is None:
            output = RegisterTensor.create(dtype=x.dtype, shape=shape)
        else:
            if shape is not None and not same_list(shape, output.shape):
                raise InstructionError(
                    f"Shape mismatch: expected {output.shape}, but got {shape} for output of load_global"
                )
        if dims is None:
            assert len(x.shape) == len(output.shape)
            dims = range(len(x.shape))
        inst = LoadGlobalInst.create(x=x, offsets=[as_expr(ofs) for ofs in offsets], dims=dims, output=output)
        self.append(inst)
        return inst.register_output

    def store_global(
        self,
        dst: GlobalTensor,
        src: RegisterTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        if dims is None:
            assert len(dst.shape) == len(src.shape)
            dims = list(range(len(dst.shape)))
        inst = StoreGlobalInst.create(dst=dst, x=src, offsets=[as_expr(ofs) for ofs in offsets], dims=dims)
        self.append(inst)

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
        shape: Sequence[int],
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
        layout: Optional[RegisterLayout] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if out is None:
            out = RegisterTensor.create(dtype=dtype, shape=shape, optional_layout=layout)
        inst = LoadGlobalGenericInst.create(ptr=ptr, f_offset=f_offset, f_mask=f_mask, output=out)
        self.append(inst)
        return inst.register_output

    def store_shared_generic(
        self,
        x: RegisterTensor,
        *,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
    ) -> None:
        inst = StoreSharedGenericInst.create(x=x, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        self.append(inst)

    def load_shared_generic(
        self,
        *,
        dtype: DataType,
        layout: RegisterLayout,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if out is None:
            out = RegisterTensor.create(dtype=dtype, shape=layout.shape, optional_layout=layout)
        inst = LoadSharedGenericInst.create(ptr=ptr, f_offset=f_offset, f_mask=f_mask, output=out)
        self.append(inst)
        return inst.register_output

    # semaphore
    def lock_semaphore(self, semaphore: Expr, value: Expr | int) -> None:
        if isinstance(value, int):
            value = as_expr(value)
        inst = LockSemaphoreInst.create(semaphore=semaphore, value=value)
        self.append(inst)

    def release_semaphore(self, semaphore: Expr, value: Expr | int) -> None:
        if isinstance(value, int):
            value = as_expr(value)
        inst = ReleaseSemaphoreInst.create(semaphore=semaphore, value=value)
        self.append(inst)

    # control operations

    def cluster_sync(self) -> None:
        inst = ClusterSyncThreadsInst.create()
        self.append(inst)

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

    # barrier
    def allocate_barrier(self, counts: Sequence[Expr | int | None]) -> RegisterTensor:
        counts_expr = [as_expr(c) if c is not None else None for c in counts]
        inst = AllocBarrierInst.create(counts=counts_expr)
        self.append(inst)
        return inst.register_output

    def arrive_barrier(self, barrier: Expr | RegisterTensor) -> None:
        if isinstance(barrier, RegisterTensor):
            barrier = self.tensor_item_value(barrier)
        inst = ArriveBarrierInst.create(barrier=barrier)
        self.append(inst)

    def arrive_remote_barrier(self, barrier: Expr | RegisterTensor, remote_block: Expr | int) -> None:
        if isinstance(barrier, RegisterTensor):
            barrier = self.tensor_item_value(barrier)
        inst = ArriveRemoteBarrierInst.create(barrier=barrier, remote_block=remote_block)
        self.append(inst)

    def wait_barrier(self, barrier: Expr | RegisterTensor, phase: Expr | int | RegisterTensor) -> None:
        if isinstance(barrier, RegisterTensor):
            barrier = self.tensor_item_value(barrier)
        if isinstance(phase, RegisterTensor):
            phase = self.tensor_item_value(phase)
        elif isinstance(phase, int):
            phase = uint32(phase)
        inst = WaitBarrierInst.create(barrier=barrier, phase=phase)
        self.append(inst)

    def fence_proxy_copy_async(self):
        inst = FenceProxyCopyAsync.create()
        self.append(inst)

    def cluster_launch_control_try_cancel(self, response: SharedTensor, mbarrier: Expr | RegisterTensor) -> None:
        if isinstance(mbarrier, RegisterTensor):
            mbarrier = self.tensor_item_value(mbarrier)
        inst = ClusterLaunchControlTryCancelInst.create(response=response, mbarrier=mbarrier)
        self.append(inst)

    def cluster_launch_control_is_canceled(self, response: RegisterTensor) -> RegisterTensor:
        inst = ClusterLaunchControlIsCanceledInst.create(response=response)
        self.append(inst)
        return inst.register_output

    def cluster_launch_control_get_first_cta(self, response: RegisterTensor) -> RegisterTensor:
        inst = ClusterLaunchControlGetFirstCtaInst.create(response=response)
        self.append(inst)
        return inst.register_output

    # tmem tensor (tcgen05)
    def tcgen05_alloc(self, dtype: DataType, shape: Sequence[int], cta_group: int) -> TMemoryTensor:
        inst = Tcgen05AllocInst.create(dtype=dtype, shape=shape, cta_group=cta_group)
        self.append(inst)
        return inst.tmemory_output

    def tcgen05_dealloc(self, tmem: TMemoryTensor) -> None:
        inst = Tcgen05DeallocInst.create(tmem)
        self.append(inst)

    def tcgen05_relinquish_alloc_permit(self, cta_group: int) -> None:
        inst = Tcgen05RelinquishAllocPermitInst.create(cta_group)
        self.append(inst)

    def tcgen05_slice(self, tmem: TMemoryTensor, offsets: Sequence[int], slice_shape: Sequence[int]) -> TMemoryTensor:
        if any(not isinstance(ofs, int) for ofs in offsets):
            raise InstructionError(f"All offsets must be integer constants, but got {offsets}")
        if len(offsets) != 2:
            raise InstructionError(f"The length of offsets must be 2, but got {len(offsets)}")
        if len(slice_shape) != 2:
            raise InstructionError(f"The length of slice_shape must be 2, but got {len(slice_shape)}")
        inst = Tcgen05SliceInst.create(tmem=tmem, offsets=offsets, shape=slice_shape)
        self.append(inst)
        return inst.tmemory_output

    def tcgen05_view(self, tmem: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> TMemoryTensor:
        if len(shape) != 2:
            raise InstructionError(f"The length of shape must be 2, but got {len(shape)}")
        inst = Tcgen05ViewInst.create(tmem=tmem, dtype=dtype, shape=shape)
        self.append(inst)
        return inst.tmemory_output

    def tcgen05_load(self, tmem: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]) -> RegisterTensor:
        if any(not isinstance(ofs, int) for ofs in offsets):
            raise InstructionError(f"All offsets must be integer constants, but got {offsets}")
        if len(offsets) != 2:
            raise InstructionError(f"The length of offsets must be 2, but got {len(offsets)}")
        inst = Tcgen05LoadInst.create(tmem=tmem, offsets=offsets, shape=shape)
        self.append(inst)
        return inst.register_output

    def tcgen05_store(self, tmem: TMemoryTensor, src: RegisterTensor, offsets: Sequence[int]) -> None:
        if any(not isinstance(ofs, int) for ofs in offsets):
            raise InstructionError(f"All offsets must be integer constants, but got {offsets}")
        if len(offsets) != 2:
            raise InstructionError(f"The length of offsets must be 2, but got {len(offsets)}")
        inst = Tcgen05StoreInst.create(tmem=tmem, src=src, offsets=offsets)
        self.append(inst)

    def tcgen05_wait_load(self) -> None:
        inst = Tcgen05WaitInst.create(wait_load=True, wait_store=False)
        self.append(inst)

    def tcgen05_wait_store(self) -> None:
        inst = Tcgen05WaitInst.create(wait_load=False, wait_store=True)
        self.append(inst)

    def tcgen05_copy(self, src: SharedTensor, dst: TMemoryTensor) -> None:
        inst = Tcgen05CopyInst.create(src=src, dst=dst)
        self.append(inst)

    def tcgen05_commit(self, mbarrier: Expr | RegisterTensor, cta_mask: Optional[int] = None) -> None:
        if isinstance(mbarrier, RegisterTensor):
            mbarrier = self.tensor_item_value(mbarrier)
        inst = Tcgen05CommitInst.create(mbarrier=mbarrier, cta_mask=cta_mask)
        self.append(inst)

    def tcgen05_mma_ss(self, a: SharedTensor, b: SharedTensor, d: TMemoryTensor) -> None:
        inst = Tcgen05MmaSSInst.create(a=a, b=b, d=d)
        self.append(inst)

    def tcgen05_mma_ts(self, a: TMemoryTensor, b: SharedTensor, d: TMemoryTensor) -> None:
        inst = Tcgen05MmaTSInst.create(a=a, b=b, d=d)
        self.append(inst)

    # annotations
    def annotate_layout(self, tensor: RegisterTensor | SharedTensor, layout: RegisterLayout | SharedLayout) -> None:
        inst = AnnotateLayoutInst.create(tensor=tensor, layout=layout)
        self.append(inst)
