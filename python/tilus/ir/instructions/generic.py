from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Sequence, Union

from hidet.ir import primitives
from hidet.ir.dtypes import DataType, boolean, i32
from hidet.ir.expr import Expr, Var, as_expr
from hidet.ir.tools import rewrite

from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.inst import Instruction
from tilus.ir.layout import RegisterLayout
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor


@dataclass(frozen=True, eq=False)
class AssignInst(Instruction):
    @staticmethod
    def create(output: Tensor, x: Tensor) -> AssignInst:
        return AssignInst(output=output, inputs=(x,))


@dataclass(frozen=True, eq=False)
class AllocateRegisterInst(Instruction):
    axes: Optional[tuple[Var, ...]]
    init: Optional[Expr]

    @staticmethod
    def create(output: RegisterTensor, f_init: Optional[Callable[[Sequence[Var]], Expr]]) -> AllocateRegisterInst:
        if f_init is not None:
            axes = tuple(index_vars(num_vars=len(output.shape)))
            init = f_init(axes)
        else:
            axes = None
            init = None
        return AllocateRegisterInst(output=output, inputs=tuple(), axes=axes, init=init)


@dataclass(frozen=True, eq=False)
class LoadGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    slice_dims: tuple[int, ...]

    @staticmethod
    def create(
        x: GlobalTensor, offsets: Sequence[Expr], slice_dims: Sequence[int], output: RegisterTensor
    ) -> LoadGlobalInst:
        return LoadGlobalInst(output=output, inputs=(x,), offsets=tuple(offsets), slice_dims=tuple(slice_dims))


@dataclass(frozen=True, eq=False)
class StoreGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    slice_dims: tuple[int, ...]

    @staticmethod
    def create(dst: GlobalTensor, x: RegisterTensor, offsets: Sequence[Expr], dims: Sequence[int]) -> StoreGlobalInst:
        return StoreGlobalInst(output=None, inputs=(dst, x), offsets=tuple(offsets), slice_dims=tuple(dims))


@dataclass(frozen=True, eq=False)
class LoadSharedInst(Instruction):
    @staticmethod
    def create(x: SharedTensor, output: RegisterTensor) -> LoadSharedInst:
        return LoadSharedInst(output=output, inputs=(x,))


@dataclass(frozen=True, eq=False)
class StoreSharedInst(Instruction):
    @staticmethod
    def create(dst: SharedTensor, src: RegisterTensor) -> StoreSharedInst:
        return StoreSharedInst(output=None, inputs=(dst, src))


@dataclass(frozen=True, eq=False)
class SharedSliceInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: Optional[tuple[int, ...]]

    @staticmethod
    def create(
        tensor: SharedTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
        shape: Sequence[int],
    ) -> SharedSliceInst:
        output = SharedTensor.create(dtype=tensor.dtype, shape=shape)
        return SharedSliceInst(
            output=output,
            inputs=(tensor,),
            offsets=tuple(offsets),
            dims=tuple(dims) if len(dims) < len(tensor.shape) else None,
        )


@dataclass(frozen=True, eq=False)
class LoadGlobalGenericInst(Instruction):
    ptr: Var
    axes: tuple[Var, ...]
    offset: Expr
    mask: Expr

    @staticmethod
    def create(
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]],
        output: RegisterTensor,
    ) -> LoadGlobalGenericInst:
        axes = tuple(index_vars(num_vars=len(output.layout.shape)))
        offset = as_expr(f_offset(axes))
        mask = as_expr(f_mask(axes)) if f_mask is not None else boolean.true
        return LoadGlobalGenericInst(output=output, inputs=tuple(), ptr=ptr, axes=axes, offset=offset, mask=mask)


@dataclass(frozen=True, eq=False)
class StoreGlobalGenericInst(Instruction):
    ptr: Var
    axes: tuple[Var, ...]
    offset: Expr
    mask: Expr

    @staticmethod
    def create(
        x: RegisterTensor,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
    ) -> StoreGlobalGenericInst:
        axes = tuple(index_vars(num_vars=len(x.layout.shape)))
        offset = as_expr(f_offset(axes))
        mask = as_expr(f_mask(axes)) if f_mask is not None else boolean.true
        return StoreGlobalGenericInst(output=None, inputs=(x,), ptr=ptr, axes=axes, offset=offset, mask=mask)


@dataclass(frozen=True, eq=False)
class LoadSharedGenericInst(Instruction):
    ptr: Var
    axes: tuple[Var, ...]
    offset: Expr
    mask: Expr

    @staticmethod
    def create(
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]],
        output: RegisterTensor,
    ) -> LoadSharedGenericInst:
        axes = tuple(index_vars(num_vars=len(output.layout.shape)))
        offset = as_expr(f_offset(axes))
        mask = as_expr(f_mask(axes)) if f_mask is not None else boolean.true
        return LoadSharedGenericInst(output=output, inputs=tuple(), ptr=ptr, axes=axes, offset=offset, mask=mask)


@dataclass(frozen=True, eq=False)
class StoreSharedGenericInst(Instruction):
    ptr: Var
    axes: tuple[Var, ...]
    offset: Expr
    mask: Expr

    @staticmethod
    def create(
        x: RegisterTensor,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
    ) -> StoreSharedGenericInst:
        axes = tuple(index_vars(num_vars=len(x.layout.shape)))
        offset = as_expr(f_offset(axes))
        mask = as_expr(f_mask(axes)) if f_mask is not None else boolean.true
        return StoreSharedGenericInst(output=None, inputs=(x,), ptr=ptr, axes=axes, offset=offset, mask=mask)


@dataclass(frozen=True, eq=False)
class CastInst(Instruction):
    @staticmethod
    def create(
        x: RegisterTensor,
        output: RegisterTensor,
    ) -> CastInst:
        return CastInst(output=output, inputs=(x,))


@dataclass(frozen=True, eq=False)
class ElementwiseUnaryBaseInst(Instruction):
    def f_compute(self, arg: Var) -> Expr:
        raise NotImplementedError("f_compute should be implemented in subclasses")


@dataclass(frozen=True, eq=False)
class ElementwiseUnaryInst(ElementwiseUnaryBaseInst):
    arg: Var
    value: Expr

    @staticmethod
    def create(x: RegisterTensor, f_compute: Callable[[Var], Expr], output: RegisterTensor) -> ElementwiseUnaryInst:
        arg = Var("x", type=x.dtype)
        value = f_compute(arg)
        return ElementwiseUnaryInst(output=output, inputs=(x,), arg=arg, value=value)

    def f_compute(self, arg: Var) -> Expr:
        return rewrite(self.value, {self.arg: arg})


@dataclass(frozen=True, eq=False)
class NegInst(ElementwiseUnaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, output: RegisterTensor) -> NegInst:
        return NegInst(output=output, inputs=(x,))

    def f_compute(self, arg: Var) -> Expr:
        return -arg


@dataclass(frozen=True, eq=False)
class AbsInst(ElementwiseUnaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, output: RegisterTensor) -> AbsInst:
        return AbsInst(output=output, inputs=(x,))

    def f_compute(self, arg: Var) -> Expr:
        return primitives.abs(arg)


@dataclass(frozen=True, eq=False)
class ClipInst(ElementwiseUnaryBaseInst):
    min: Expr
    max: Expr

    @staticmethod
    def create(x: RegisterTensor, min: Expr | int | float, max: Expr | int | float, output: RegisterTensor) -> ClipInst:
        min = x.dtype(min)
        max = x.dtype(max)
        return ClipInst(output=output, inputs=(x,), min=min, max=max)

    def f_compute(self, arg: Var) -> Expr:
        return primitives.min(primitives.max(arg, self.min), self.max)


@dataclass(frozen=True, eq=False)
class ElementwiseBinaryBaseInst(Instruction):
    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        raise NotImplementedError("f_compute should be implemented in subclasses")


@dataclass(frozen=True, eq=False)
class ElementwiseBinaryInst(ElementwiseBinaryBaseInst):
    args: tuple[Var, Var]
    value: Expr

    @staticmethod
    def create(
        x: RegisterTensor, y: RegisterTensor, f_compute: Callable[[Var, Var], Expr], output: RegisterTensor
    ) -> ElementwiseBinaryInst:
        lhs = Var("x", type=x.dtype)
        rhs = Var("y", type=y.dtype)
        value = f_compute(lhs, rhs)
        return ElementwiseBinaryInst(output=output, inputs=(x, y), args=(lhs, rhs), value=value)

    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        return rewrite(self.value, {self.args[0]: lhs, self.args[1]: rhs})


@dataclass(frozen=True, eq=False)
class AddInst(ElementwiseBinaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, y: RegisterTensor, output: RegisterTensor) -> AddInst:
        return AddInst(output=output, inputs=(x, y))

    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        return lhs + rhs


@dataclass(frozen=True, eq=False)
class SubInst(ElementwiseBinaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, y: RegisterTensor, output: RegisterTensor) -> SubInst:
        return SubInst(output=output, inputs=(x, y))

    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        return lhs - rhs


@dataclass(frozen=True, eq=False)
class MulInst(ElementwiseBinaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, y: RegisterTensor, output: RegisterTensor) -> MulInst:
        return MulInst(output=output, inputs=(x, y))

    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        return lhs * rhs


@dataclass(frozen=True, eq=False)
class DivInst(ElementwiseBinaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, y: RegisterTensor, output: RegisterTensor) -> DivInst:
        return DivInst(output=output, inputs=(x, y))

    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        return lhs / rhs


@dataclass(frozen=True, eq=False)
class ModInst(ElementwiseBinaryBaseInst):
    @staticmethod
    def create(x: RegisterTensor, y: RegisterTensor, output: RegisterTensor) -> ModInst:
        return ModInst(output=output, inputs=(x, y))

    def f_compute(self, lhs: Var, rhs: Var) -> Expr:
        return lhs % rhs


@dataclass(frozen=True, eq=False)
class WhereInst(Instruction):
    @staticmethod
    def create(cond: RegisterTensor, x: RegisterTensor, y: RegisterTensor, output: RegisterTensor) -> WhereInst:
        return WhereInst(output=output, inputs=(cond, x, y))


@dataclass(frozen=True, eq=False)
class RepeatInst(Instruction):
    @staticmethod
    def create(x: RegisterTensor, output: RegisterTensor) -> RepeatInst:
        return RepeatInst(output=output, inputs=(x,))


@dataclass(frozen=True, eq=False)
class RepeatInterleaveInst(Instruction):
    @staticmethod
    def create(x: RegisterTensor, output: RegisterTensor) -> RepeatInterleaveInst:
        return RepeatInterleaveInst(output=output, inputs=(x,))


@dataclass(frozen=True, eq=False)
class FormatPrintInst(Instruction):
    cond: Expr
    fstring: str
    expressions: tuple[Expr, ...]

    @staticmethod
    def create(cond: Expr, fstring: str, expressions_: Sequence[Expr | float | int | str] = tuple()) -> FormatPrintInst:
        expressions = [as_expr(e) for e in expressions_]
        return FormatPrintInst(output=None, inputs=(), cond=cond, fstring=fstring, expressions=tuple(expressions))


@dataclass(frozen=True, eq=False)
class PrintTensorInst(Instruction):
    cond: Expr
    msg: str
    fmt: Optional[str]

    @staticmethod
    def create(x: Tensor, cond: Expr, msg: str, fmt: Optional[str] = None) -> PrintTensorInst:
        return PrintTensorInst(output=None, inputs=(x,), cond=cond, msg=msg, fmt=fmt)


@dataclass(frozen=True, eq=False)
class ShuffleBaseInst(Instruction):
    mask: int
    delta: int
    width: int


@dataclass(frozen=True, eq=False)
class ShuffleDownInst(ShuffleBaseInst):
    pass


@dataclass(frozen=True, eq=False)
class ShuffleUpInst(ShuffleBaseInst):
    pass


@dataclass(frozen=True, eq=False)
class ReduceInst(Instruction):
    dim: int
    op: str
    keepdim: bool
    VALID_OPS: ClassVar[tuple[str, ...]] = ("sum", "max", "min")

    @staticmethod
    def create(
        x: RegisterTensor,
        dim: int,
        keepdim: bool,
        op: str,
        output: RegisterTensor,
    ) -> ReduceInst:
        assert op in ReduceInst.VALID_OPS
        return ReduceInst(output=output, inputs=(x,), dim=dim, keepdim=keepdim, op=op)


@dataclass(frozen=True, eq=False)
class ViewInst(Instruction):
    local_offset: Expr

    @staticmethod
    def create(
        x: RegisterTensor,
        *,
        layout: Optional[RegisterLayout] = None,
        dtype: Optional[DataType] = None,
        local_offset: Union[Expr, int] = 0,
    ) -> ViewInst:
        dtype = dtype if dtype else x.dtype
        layout = layout if layout else x.layout
        output = RegisterTensor.create(dtype=dtype, optional_layout=layout)
        return ViewInst(output=output, inputs=(x,), local_offset=i32(local_offset))


@dataclass(frozen=True, eq=False)
class SqueezeInst(Instruction):
    dims: tuple[int, ...]

    @staticmethod
    def create(
        x: RegisterTensor,
        *,
        dims: Sequence[int] | int,
        out: Optional[RegisterTensor] = None,
    ) -> SqueezeInst:
        if isinstance(dims, int):
            dims = [dims]
        if out is None:
            from tilus.ir.layout.register_layout_ops import squeeze

            out = RegisterTensor.create(dtype=x.dtype, optional_layout=squeeze(x.layout, dims))
        return SqueezeInst(output=out, inputs=(x,), dims=tuple(dims))


@dataclass(frozen=True, eq=False)
class UnsqueezeInst(Instruction):
    dims: tuple[int, ...]

    @staticmethod
    def create(
        x: RegisterTensor,
        *,
        dims: Sequence[int] | int,
        out: Optional[RegisterTensor] = None,
    ) -> UnsqueezeInst:
        if isinstance(dims, int):
            dims = [dims]
        if out is None:
            from tilus.ir.layout.register_layout_ops import unsqueeze

            out = RegisterTensor.create(dtype=x.dtype, optional_layout=unsqueeze(x.layout, dims))
        return UnsqueezeInst(output=out, inputs=(x,), dims=tuple(dims))


@dataclass(frozen=True, eq=False)
class TransposeInst(Instruction):
    @staticmethod
    def create(x: RegisterTensor, out: Optional[RegisterTensor] = None) -> TransposeInst:
        assert len(x.shape) == 2
        if out is None:
            from tilus.ir.layout.register_layout_ops import permute

            out = RegisterTensor.create(dtype=x.dtype, optional_layout=permute(x.layout, [1, 0]))
        return TransposeInst(output=out, inputs=(x,))


@dataclass(frozen=True, eq=False)
class AllocateSharedInst(Instruction):
    @staticmethod
    def create(output: SharedTensor) -> AllocateSharedInst:
        return AllocateSharedInst(output=output, inputs=())


@dataclass(frozen=True, eq=False)
class AllocateGlobalInst(Instruction):
    require_clean: bool

    @staticmethod
    def create(output: GlobalTensor, require_clean: bool) -> AllocateGlobalInst:
        return AllocateGlobalInst(output=output, inputs=(), require_clean=require_clean)

    def with_output(self, global_output: GlobalTensor) -> AllocateGlobalInst:
        return dataclasses.replace(self, output=global_output)


@dataclass(frozen=True, eq=False)
class GlobalViewInst(Instruction):
    ptr: Expr

    @staticmethod
    def create(output: GlobalTensor, ptr: Expr) -> GlobalViewInst:
        return GlobalViewInst(output=output, inputs=(), ptr=ptr)


@dataclass(frozen=True, eq=False)
class FreeSharedInst(Instruction):
    @staticmethod
    def create(tensor: SharedTensor) -> FreeSharedInst:
        return FreeSharedInst(output=None, inputs=(tensor,))


@dataclass(frozen=True, eq=False)
class SyncThreadsInst(Instruction):
    @staticmethod
    def create() -> SyncThreadsInst:
        return SyncThreadsInst(output=None, inputs=())


@dataclass(frozen=True, eq=False)
class SyncReduceThreadsInst(Instruction):
    AND: ClassVar[str] = "and"
    OR: ClassVar[str] = "or"
    reduce_op: str
    var: Var
    reduce_value: Expr

    @staticmethod
    def create(reduce_op: str, var_hint: str, reduce_value: Expr) -> SyncReduceThreadsInst:
        var = Var(var_hint, type=boolean)
        return SyncReduceThreadsInst(output=None, inputs=(), reduce_op=reduce_op, var=var, reduce_value=reduce_value)


@dataclass(frozen=True, eq=False)
class ExitInst(Instruction):
    @staticmethod
    def create() -> ExitInst:
        return ExitInst(output=None, inputs=())


@dataclass(frozen=True, eq=False)
class NopInst(Instruction):
    @staticmethod
    def create() -> NopInst:
        return NopInst(output=None, inputs=())
