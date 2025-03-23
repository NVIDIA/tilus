from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from hidet.ir.dtypes import bf16, boolean, f16, f32, i8, i32
from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType
from tilus.extensions.hidet.ir.expr import as_expr, index_vars
from tilus.ir.layout import RegisterLayout, column_repeat, column_spatial, repeat, spatial
from tilus.ir.node import IRNode
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedLayout, SharedTensor, Tensor


@dataclass(frozen=True, eq=False)
class Instruction(IRNode):
    output: Optional[Tensor]
    inputs: tuple[Tensor, ...]

    @property
    def shared_output(self) -> SharedTensor:
        assert isinstance(self.output, SharedTensor)
        return self.output

    @property
    def register_output(self) -> RegisterTensor:
        assert isinstance(self.output, RegisterTensor)
        return self.output

    @property
    def register_or_shared_output(self) -> SharedTensor | RegisterTensor:
        assert isinstance(self.output, SharedTensor) or isinstance(self.output, RegisterTensor)
        return self.output

    @property
    def global_output(self) -> GlobalTensor:
        assert isinstance(self.output, GlobalTensor)
        return self.output

    @property
    def register_input(self) -> RegisterTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, RegisterTensor)
        return x

    @property
    def shared_input(self) -> SharedTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, SharedTensor)
        return x

    @property
    def attributes(self) -> Mapping[str, Any]:
        attrs = {}
        for k, v in self.__dict__.items():
            if k in ["output", "inputs"]:
                continue
            attrs[k] = v
        return attrs


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
            axes = tuple(index_vars(num_vars=len(output.layout.shape)))
            init = f_init(axes)
        else:
            axes = None
            init = None
        return AllocateRegisterInst(output=output, inputs=tuple(), axes=axes, init=init)


@dataclass(frozen=True, eq=False)
class LoadGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]

    @staticmethod
    def create(x: GlobalTensor, offsets: Sequence[Expr], dims: Sequence[int], output: RegisterTensor) -> LoadGlobalInst:
        return LoadGlobalInst(output=output, inputs=(x,), offsets=tuple(offsets), dims=tuple(dims))


@dataclass(frozen=True, eq=False)
class StoreGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]

    @staticmethod
    def create(dst: GlobalTensor, x: RegisterTensor, offsets: Sequence[Expr], dims: Sequence[int]) -> StoreGlobalInst:
        return StoreGlobalInst(output=None, inputs=(dst, x), offsets=tuple(offsets), dims=tuple(dims))


@dataclass(frozen=True, eq=False)
class LoadSharedInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]

    @staticmethod
    def create(
        x: SharedTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
        output: RegisterTensor,
    ) -> LoadSharedInst:
        return LoadSharedInst(output=output, inputs=(x,), offsets=tuple(offsets), dims=tuple(dims))


@dataclass(frozen=True, eq=False)
class StoreSharedInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]

    @staticmethod
    def create(
        dst: SharedTensor,
        src: RegisterTensor,
        offsets: Sequence[Expr],
        dims: Sequence[int],
    ) -> StoreSharedInst:
        return StoreSharedInst(output=None, inputs=(dst, src), offsets=tuple(offsets), dims=tuple(dims))


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
class ElementwiseUnaryInst(Instruction):
    VALID_OPS: ClassVar[tuple[str, ...]] = ("relu",)
    op: str

    @staticmethod
    def create(x: RegisterTensor, op: str, output: RegisterTensor) -> ElementwiseUnaryInst:
        return ElementwiseUnaryInst(output=output, inputs=(x,), op=op)


@dataclass(frozen=True, eq=False)
class ElementwiseBinaryInst(Instruction):
    VALID_OPS: ClassVar[tuple[str, ...]] = ("+", "-", "*", "/", "%")
    op: str

    @staticmethod
    def create(x: RegisterTensor, y: RegisterTensor, op: str, output: RegisterTensor) -> ElementwiseBinaryInst:
        return ElementwiseBinaryInst(output=output, inputs=(x, y), op=op)


@dataclass(frozen=True, eq=False)
class BroadcastElementwiseBinaryInst(Instruction):
    VALID_OPS: ClassVar[tuple[str, ...]] = ("+", "-", "*", "/", "%")
    op: str
    s: Expr
    tensor_left: bool

    @staticmethod
    def create(tensor: RegisterTensor, scalar: Expr, op: str, tensor_left: bool) -> BroadcastElementwiseBinaryInst:
        return BroadcastElementwiseBinaryInst(output=tensor, inputs=(tensor,), op=op, s=scalar, tensor_left=tensor_left)


@dataclass(frozen=True, eq=False)
class MmaDotInst(Instruction):
    mma_inst: str
    warp_spatial: Tuple[int, int, int]
    warp_repeat: Tuple[int, int, int]

    @staticmethod
    def create(
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        mma_inst: str,
        warp_spatial: Sequence[int],
        warp_repeat: Sequence[int],
        output: RegisterTensor,
    ) -> MmaDotInst:
        if len(warp_spatial) == 2:
            warp_spatial_ = (warp_spatial[0], warp_spatial[1], 1)
        elif len(warp_spatial) == 3:
            warp_spatial_ = (warp_spatial[0], warp_spatial[1], warp_spatial[2])
        else:
            raise ValueError("warp_spatial must have length 2 or 3")
        if len(warp_repeat) == 3:
            warp_repeat_ = (warp_repeat[0], warp_repeat[1], warp_repeat[2])
        else:
            raise ValueError("warp_repeat must have length 3")
        return MmaDotInst(
            output=output, inputs=(a, b, c), mma_inst=mma_inst, warp_spatial=warp_spatial_, warp_repeat=warp_repeat_
        )


@dataclass(frozen=True, eq=False)
class SimtDotInst(Instruction):
    warp_spatial: Tuple[int, int, int]
    warp_repeat: Tuple[int, int, int]
    thread_spatial: Tuple[int, int]
    thread_repeat: Tuple[int, int]

    @staticmethod
    def create(
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        thread_spatial: Tuple[int, int],
        thread_repeat: Tuple[int, int],
        output: Optional[RegisterTensor] = None,
    ) -> SimtDotInst:
        if output is None:
            output = RegisterTensor.create(c.dtype, c.layout)
        return SimtDotInst(
            output=output,
            inputs=(a, b, c),
            warp_spatial=warp_spatial,
            warp_repeat=warp_repeat,
            thread_spatial=thread_spatial,
            thread_repeat=thread_repeat,
        )


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
class PrintValueInst(Instruction):
    cond: Expr
    msg: str
    fmt: Optional[str]

    @staticmethod
    def create(x: Tensor, cond: Expr, msg: str, fmt: Optional[str] = None) -> PrintValueInst:
        return PrintValueInst(output=None, inputs=(x,), cond=cond, msg=msg, fmt=fmt)


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
        output = RegisterTensor.create(dtype=dtype, layout=layout)
        return ViewInst(output=output, inputs=(x,), local_offset=i32(local_offset))


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
class CopyAsyncInst(Instruction):
    ptr: Var
    axes: List[Var]
    offset: Expr
    mask: Optional[Expr]
    evict: Optional[str]

    @staticmethod
    def supports(
        dtype: DataType,
        shared_layout: SharedLayout,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]],
        divisibility: Dict[Var, int],
    ) -> bool:
        raise NotImplementedError()

    @staticmethod
    def create(
        dst: SharedTensor,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]],
        evict: Optional[str] = None,
    ) -> CopyAsyncInst:
        axes = index_vars(len(dst.shape))
        offset = f_offset(axes)
        mask = f_mask(axes) if f_mask else None
        return CopyAsyncInst(output=None, inputs=(dst,), ptr=ptr, axes=axes, offset=offset, mask=mask, evict=evict)


@dataclass(frozen=True, eq=False)
class CopyAsyncCommitGroupInst(Instruction):
    @staticmethod
    def create() -> CopyAsyncCommitGroupInst:
        return CopyAsyncCommitGroupInst(output=None, inputs=())


@dataclass(frozen=True, eq=False)
class CopyAsyncWaitGroupInst(Instruction):
    n: Expr

    @staticmethod
    def create(n: Expr) -> CopyAsyncWaitGroupInst:
        return CopyAsyncWaitGroupInst(output=None, inputs=(), n=n)


@dataclass(frozen=True, eq=False)
class CopyAsyncWaitAllInst(Instruction):
    @staticmethod
    def create() -> CopyAsyncWaitAllInst:
        return CopyAsyncWaitAllInst(output=None, inputs=())


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
class LoadMatrixInst(Instruction):
    LDMATRIX_CONFIGS = [
        # (dtype bytes, trans, ldmatrix_layout)
        (1, False, spatial(8, 4).repeat(1, 4)),
        # (1, True, column_spatial(4, 8).repeat(4, 1)), # ldmatrix does not support this case
        (2, False, spatial(8, 4).repeat(1, 2)),
        (2, True, column_spatial(4, 8).repeat(2, 1)),
    ]

    offsets: List[Expr]

    @staticmethod
    def create(
        src: SharedTensor, register_layout: RegisterLayout, offsets: List[Expr], output: Optional[RegisterTensor] = None
    ) -> LoadMatrixInst:
        if output is None:
            output = RegisterTensor.create(dtype=src.dtype, layout=register_layout)
        else:
            assert output.dtype == src.dtype and output.layout.quick_equal(register_layout)
        return LoadMatrixInst(output=output, inputs=(src,), offsets=offsets)


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


@dataclass(frozen=True, eq=False)
class MmaConfig:
    name: str
    m: int
    n: int
    k: int
    vec_k: int
    la: RegisterLayout
    lb: RegisterLayout
    lc: RegisterLayout
    operand_type: DataType
    acc_type: DataType

    def __hash__(self):
        return hash((MmaConfig, self.name))

    def __eq__(self, other):
        return isinstance(other, MmaConfig) and self.name == other.name

    def hidet_mma_config(self):
        from hidet.ir.primitives.cuda.mma import MmaConfig

        v_pos = self.name.find("v")
        under_pos = self.name.find("_", v_pos)
        hidet_config_name = self.name[:v_pos] + self.name[under_pos:]

        return getattr(MmaConfig, hidet_config_name)()

    @staticmethod
    def m16n8k16_f16_f16(vec_k: int = 1) -> MmaConfig:
        return MmaConfig(
            name="m16n8k16v{}_f16_f16".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            operand_type=f16,
            acc_type=f16,
        )

    @staticmethod
    @lru_cache()
    def m16n8k16_f16_f32(vec_k: int = 1) -> MmaConfig:
        return MmaConfig(
            name="m16n8k16v{}_f16_f32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            operand_type=f16,
            acc_type=f32,
        )

    @staticmethod
    @lru_cache()
    def m16n8k16_bf16_f32(vec_k: int = 1) -> MmaConfig:
        return MmaConfig(
            name="m16n8k16v{}_bf16_f32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            operand_type=bf16,
            acc_type=f32,
        )

    @staticmethod
    @lru_cache()
    def m8n8k16_i8_i32(vec_k: int = 1) -> MmaConfig:
        return MmaConfig(
            name="m8n8k16v{}_i8_i32".format(vec_k),
            m=8,
            n=8,
            k=16,
            vec_k=vec_k,
            la=spatial(8, 4).repeat(1, 4),
            lb=column_spatial(4, 8).repeat(4, 1),
            lc=spatial(8, 4).repeat(1, 2),
            operand_type=i8,
            acc_type=i32,
        )

    @staticmethod
    @lru_cache()
    def all() -> dict[str, MmaConfig]:
        config_list = []
        for vec_k in [1, 2, 3, 4]:
            config_list.append(MmaConfig.m16n8k16_f16_f32(vec_k))
            config_list.append(MmaConfig.m16n8k16_f16_f16(vec_k))
            config_list.append(MmaConfig.m16n8k16_bf16_f32(vec_k))
            config_list.append(MmaConfig.m8n8k16_i8_i32(vec_k))
        return {config.name: config for config in config_list}

    @staticmethod
    def from_name(name: str) -> MmaConfig:
        return MmaConfig.all()[name]
