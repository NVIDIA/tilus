from __future__ import annotations
from typing import List, Union, Optional, Callable, Dict, Any, Tuple, Sequence, Mapping, ClassVar
from dataclasses import dataclass
from functools import lru_cache
from hidet.ir.expr import Var, Expr
from hidet.ir.type import DataType, BaseType, PointerType
from hidet.ir.dtypes import bf16, f16, f32, i32, i8, boolean
from tilus.extensions.hidet.ir.expr import index_vars, convert_to_expr
from tilus.ir.layout import Layout, spatial, repeat, column_repeat, column_spatial
from tilus.ir.value import Value, RegisterValue, SharedValue, SharedLayout


@dataclass(frozen=True, eq=False)
class Instruction:
    output: Optional[Value]
    inputs: tuple[Value, ...]

    def __str__(self):
        items = []
        if self.output is not None:
            items.append(str(self.output))
        items.extend([str(t) for t in self.inputs])
        items.extend([f"{k}={v}" for k, v in self.attrs.items()])
        return "{}({})".format(self.__class__.__name__, ",\n".join(items))

    @property
    def shared_output(self) -> SharedValue:
        assert isinstance(self.output, SharedValue)
        return self.output

    @property
    def register_output(self) -> RegisterValue:
        assert isinstance(self.output, RegisterValue)
        return self.output

    @property
    def attributes(self) -> Mapping[str, Any]:
        attrs = {}
        for k, v in self.__dict__.items():
            if k in ["output", "inputs"]:
                continue
            attrs[k] = v
        return attrs

    def recreate(self, updated_output: Optional[Value], updated_inputs: List[Value], updated_attrs: Dict[str, Any]):
        # by default, all subclasses of Instruction will have __init__ accepts ([output, ]*inputs, **attrs)
        # if the subclass has different signature, it should override this method so that we can recreate the instance
        # with the updated values and attrs
        values = []
        if updated_output:
            values.append(updated_output)
        values.extend(updated_inputs)
        return type(self)(*values, **updated_attrs)  # type: ignore


@dataclass(frozen=True, eq=False)
class AssignInst(Instruction):
    @staticmethod
    def create(output: Value, x: Value):
        return AssignInst(output=output, inputs=(x,))


@dataclass(frozen=True, eq=False)
class AllocateInst(Instruction):
    init: Optional[tuple[tuple[Var, ...], Expr]]  # mapping from axes => init_value

    @staticmethod
    def create(
        dtype: DataType, layout: Layout, f_init: Optional[Callable[[Sequence[Var]], Expr]] = None
    ) -> AllocateInst:
        out = RegisterValue.create(dtype, layout)
        if f_init is not None:
            axes = tuple(index_vars(num_vars=len(layout.shape)))
            value = f_init(axes)
            init = (axes, value)
        else:
            init = None
        return AllocateInst(output=out, inputs=tuple(), init=init)


@dataclass(frozen=True, eq=False)
class AllocateScalarInst(Instruction):
    var: Var
    init: Optional[Expr]

    @staticmethod
    def create(hint: str, scalar_type: Union[DataType, PointerType], init: Optional[Expr] = None):
        var = Var(hint=hint, type=scalar_type)
        return AllocateScalarInst(output=None, inputs=tuple(), var=var, init=init)


@dataclass(frozen=True, eq=False)
class AssignScalarInst(Instruction):
    var: Var
    scalar_expr: Expr

    @staticmethod
    def create(var: Var, scalar_expr: Expr):
        return AssignScalarInst(output=None, inputs=tuple(), var=var, scalar_expr=scalar_expr)


@dataclass(frozen=True, eq=False)
class LoadGlobalInst(Instruction):
    ptr: Var
    axes: List[Var]
    offset: Expr
    mask: Expr

    @staticmethod
    def create(
        dtype: DataType,
        layout: Layout,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr | int],
        f_mask: Optional[Callable[[List[Var]], Expr | int | bool]] = None,
        out: Optional[RegisterValue] = None,
    ) -> LoadGlobalInst:
        if out is None:
            out = RegisterValue.create(dtype, layout)
        axes = index_vars(num_vars=len(layout.shape))
        offset = convert_to_expr(f_offset(axes))
        mask = convert_to_expr(f_mask(axes)) if f_mask is not None else boolean.true
        return LoadGlobalInst(output=out, inputs=tuple(), ptr=ptr, axes=axes, offset=offset, mask=mask)


@dataclass(frozen=True, eq=False)
class StoreGlobalInst(Instruction):
    ptr: Var
    axes: List[Var]
    offset: Expr
    mask: Expr

    @staticmethod
    def create(
        x: RegisterValue,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr | int],
        f_mask: Optional[Callable[[List[Var]], Expr | int | bool]] = None,
    ) -> StoreGlobalInst:
        axes = index_vars(num_vars=len(x.layout.shape))
        offset = convert_to_expr(f_offset(axes))
        mask = convert_to_expr(f_mask(axes)) if f_mask is not None else boolean.true
        return StoreGlobalInst(output=None, inputs=(x,), ptr=ptr, axes=axes, offset=offset, mask=mask)


@dataclass(frozen=True, eq=False)
class CastInst(Instruction):
    interleave_width: Optional[int]
    interleave_stride: Optional[int]
    ignore_int4b_xor: bool

    @staticmethod
    def create(
        dtype: DataType, x: RegisterValue, interleave_width=None, interleave_stride=None, ignore_int4b_xor=False
    ) -> CastInst:
        out = RegisterValue.create(dtype, x.layout)
        return CastInst(
            output=out,
            inputs=(x,),
            interleave_width=interleave_width,
            interleave_stride=interleave_stride,
            ignore_int4b_xor=ignore_int4b_xor,
        )


@dataclass(frozen=True, eq=False)
class ElementwiseUnaryInst(Instruction):
    VALID_OPS: ClassVar[tuple[str, ...]] = ("relu", "clip")
    op: str

    @staticmethod
    def create(x: RegisterValue, op: str, *, output: Optional[RegisterValue] = None) -> ElementwiseUnaryInst:
        assert op in ElementwiseUnaryInst.VALID_OPS
        if output is None:
            output = RegisterValue.create(x.dtype, x.layout)
        return ElementwiseUnaryInst(output=output, inputs=(x,), op=op)


@dataclass(frozen=True, eq=False)
class ElementwiseBinaryInst(Instruction):
    VALID_OPS: ClassVar[tuple[str, ...]] = ("+", "-", "*", "/", "%")
    op: str

    @staticmethod
    def create(x: RegisterValue, y: RegisterValue, op: str, *, output: Optional[RegisterValue]):
        assert op in ElementwiseBinaryInst.VALID_OPS
        if output is None:
            output = RegisterValue.create(x.dtype, x.layout)
        return ElementwiseBinaryInst(output=output, inputs=(x, y), op=op)


@dataclass(frozen=True, eq=False)
class BroadcastElementwiseBinaryInst(Instruction):
    VALID_OPS: ClassVar[tuple[str, ...]] = ("+", "-", "*", "/", "%")
    op: str
    s: Expr
    tensor_left: bool

    @staticmethod
    def create(
        x: Union[RegisterValue, Expr], y: Union[RegisterValue, Expr], op: str, *, output: Optional[RegisterValue]
    ):
        assert op in BroadcastElementwiseBinaryInst.VALID_OPS
        if isinstance(x, RegisterValue) and isinstance(y, Expr):
            r, s = x, y
            tensor_left = True
        elif isinstance(x, Expr) and isinstance(y, RegisterValue):
            r, s = y, x
            tensor_left = False
        else:
            assert False
        if op in ["+", "-", "*", "/", "%"]:
            out_dtype = r.dtype
        else:
            raise NotImplementedError()

        if output is None:
            output = RegisterValue.create(out_dtype, r.layout)
        return BroadcastElementwiseBinaryInst(output=output, inputs=(r,), op=op, s=s, tensor_left=tensor_left)


@dataclass(frozen=True, eq=False)
class MmaDotInst(Instruction):
    mma_inst: str
    warp_spatial: Tuple[int, int, int]
    warp_repeat: Tuple[int, int, int]

    @staticmethod
    def create(
        a: RegisterValue,
        b: RegisterValue,
        c: RegisterValue,
        mma_inst: str,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        output: Optional[RegisterValue] = None,
    ):
        if output is None:
            output = RegisterValue.create(c.dtype, c.layout)
        return MmaDotInst(
            output=output, inputs=(a, b, c), mma_inst=mma_inst, warp_spatial=warp_spatial, warp_repeat=warp_repeat
        )


@dataclass(frozen=True, eq=False)
class SimtDotInst(Instruction):
    warp_spatial: Tuple[int, int, int]
    warp_repeat: Tuple[int, int, int]
    thread_spatial: Tuple[int, int]
    thread_repeat: Tuple[int, int]

    @staticmethod
    def create(
        a: RegisterValue,
        b: RegisterValue,
        c: RegisterValue,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        thread_spatial: Tuple[int, int],
        thread_repeat: Tuple[int, int],
        output: Optional[RegisterValue] = None,
    ):
        if output is None:
            output = RegisterValue.create(c.dtype, c.layout)
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
    expressions: List[Expr]

    @staticmethod
    def create(cond: Expr, fstring: str, expressions: Sequence[Expr] = tuple()):
        return FormatPrintInst(output=None, inputs=(), cond=cond, fstring=fstring, expressions=list(expressions))


@dataclass(frozen=True, eq=False)
class PrintValueInst(Instruction):
    cond: Expr
    msg: str
    fmt: Optional[str]

    @staticmethod
    def create(x: Value, cond: Expr, msg: str, fmt: Optional[str] = None):
        return PrintValueInst(output=None, inputs=(x,), cond=cond, msg=msg, fmt=fmt)


@dataclass(frozen=True, eq=False)
class ShuffleBaseInst(Instruction):
    mask: int
    delta: int
    width: int

    # def __post_init__(self):
    #     warp_size = 32
    #     output = self.output
    #     self.dtype: DataType = output.as_register_value().dtype
    #     self.layout: Layout = output.as_register_value().layout
    #     self.num_warps: int = self.layout.num_workers // 32
    #     self.num_groups: int = max([i // self.width for i in range(self.num_warps) if self.mask & (1 << i)]) + 1
    #     self.smem_shape: Tuple[int, int, int] = (
    #         self.num_groups,
    #         self.width - self.delta,
    #         warp_size * self.layout.local_size,
    #     )
    #
    #     assert all(is_power_of_two(v) for v in [self.delta, self.width, self.num_warps])


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
        x: RegisterValue,
        *,
        layout: Optional[Layout] = None,
        dtype: Optional[DataType] = None,
        local_offset: Union[Expr, int] = 0,
    ):
        dtype = dtype if dtype else x.dtype
        layout = layout if layout else x.layout
        output = RegisterValue.create(dtype=dtype, layout=layout)
        return ViewInst(output=output, inputs=(x,), local_offset=i32(local_offset))


@dataclass(frozen=True, eq=False)
class AllocateSharedInst(Instruction):
    axes: Optional[List[Var]]
    init: Optional[Expr]

    @staticmethod
    def create(
        dtype: DataType, shared_layout: SharedLayout, f_init: Optional[Callable[[List[Var]], Expr]] = None
    ) -> AllocateSharedInst:
        out = SharedValue.create(dtype=dtype, layout=shared_layout)
        if f_init:
            axes = index_vars(num_vars=len(shared_layout.shape))
            init = f_init(axes)
        else:
            axes = None
            init = None
        return AllocateSharedInst(output=out, inputs=(), axes=axes, init=init)


@dataclass(frozen=True, eq=False)
class AllocateGlobalInst(Instruction):
    var: Var
    nbytes: Expr
    require_clean: bool

    @staticmethod
    def create(hint: str, scalar_type: BaseType, nbytes: Union[Expr, int], require_clean: bool = False):
        return AllocateGlobalInst(
            output=None,
            inputs=(),
            var=Var(hint, scalar_type),
            nbytes=nbytes if isinstance(nbytes, Expr) else i32(nbytes),
            require_clean=require_clean,
        )


@dataclass(frozen=True, eq=False)
class FreeSharedInst(Instruction):
    @staticmethod
    def create(shared_value: SharedValue):
        return FreeSharedInst(output=None, inputs=(shared_value,))


@dataclass(frozen=True, eq=False)
class ViewSharedInst(Instruction):
    indices: List[Expr]
    dtype: DataType
    layout: SharedLayout

    @staticmethod
    def create(x: SharedValue, indices: List[Expr], layout: SharedLayout, dtype: Optional[DataType] = None):
        if dtype is None:
            dtype = x.dtype
        out = SharedValue.create(dtype=dtype, layout=layout)
        return ViewSharedInst(output=out, inputs=(x,), indices=indices, dtype=dtype, layout=layout)


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
        dst: SharedValue,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]],
        evict: Optional[str] = None,
    ):
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
    def create(reduce_op: str, var_hint: str, reduce_value: Expr):
        var = Var(var_hint, type=boolean)
        return SyncReduceThreadsInst(output=None, inputs=(), reduce_op=reduce_op, var=var, reduce_value=reduce_value)


@dataclass(frozen=True, eq=False)
class LoadScalarInst(Instruction):
    var: Var
    ptr: Expr
    sync: str

    @staticmethod
    def create(ptr: Expr, sync: str = "weak"):
        from hidet.ir.tools import infer_type

        ptr_type = infer_type(ptr)
        assert isinstance(ptr_type, PointerType)
        var_type = ptr_type.base_type
        assert isinstance(var_type, DataType)
        var = Var("loaded_{}".format(var_type.name), var_type)
        return LoadScalarInst(output=None, inputs=(), var=var, ptr=ptr, sync=sync)


@dataclass(frozen=True, eq=False)
class StoreScalarInst(Instruction):
    ptr: Expr
    value: Expr
    sync: str

    @staticmethod
    def create(ptr: Expr, value: Expr, sync: str = "weak"):
        return StoreScalarInst(output=None, inputs=(), ptr=ptr, value=value, sync=sync)


@dataclass(frozen=True, eq=False)
class AtomicScalarInst(Instruction):
    ptr: Expr
    op: str
    value: Expr

    @staticmethod
    def create(ptr: Expr, op: str, value: Expr):
        return AtomicScalarInst(output=None, inputs=(), ptr=ptr, op=op, value=value)


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
        src: SharedValue, register_layout: Layout, offsets: List[Expr], output: Optional[RegisterValue] = None
    ) -> LoadMatrixInst:
        if output is None:
            output = RegisterValue.create(dtype=src.dtype, layout=register_layout)
        else:
            assert output.dtype == src.dtype and output.layout.quick_equal(register_layout)
        return LoadMatrixInst(output=output, inputs=(src,), offsets=offsets)


@dataclass(frozen=True, eq=False)
class LoadSharedInst(Instruction):
    offsets: List[Expr]

    @staticmethod
    def create(
        src: SharedValue, register_layout: Layout, offsets: List[Expr], *, output: Optional[RegisterValue] = None
    ) -> LoadSharedInst:
        if output is None:
            output = RegisterValue.create(dtype=src.dtype, layout=register_layout)
        else:
            assert output.dtype == src.dtype and output.layout.quick_equal(register_layout)
        return LoadSharedInst(output=output, inputs=(src,), offsets=offsets)


@dataclass(frozen=True, eq=False)
class StoreSharedInst(Instruction):
    offsets: List[Expr]

    @staticmethod
    def create(dst: SharedValue, src: RegisterValue, offsets: Optional[List[Expr]] = None) -> StoreSharedInst:
        if offsets is None:
            offsets = [i32.zero for _ in range(len(dst.shape))]
        return StoreSharedInst(output=None, inputs=(dst, src), offsets=offsets)


@dataclass(frozen=True, eq=False)
class ExitInst(Instruction):
    @staticmethod
    def create() -> ExitInst:
        return ExitInst(output=None, inputs=())


class MmaConfig:
    def __init__(
        self,
        name: str,
        m: int,
        n: int,
        k: int,
        vec_k: int,
        la: Layout,
        lb: Layout,
        lc: Layout,
        operand_type: DataType,
        acc_type: DataType,
    ):
        self.name: str = name
        self.m: int = m
        self.n: int = n
        self.k: int = k
        self.vec_k: int = vec_k
        self.la: Layout = la
        self.lb: Layout = lb
        self.lc: Layout = lc
        self.operand_type: DataType = operand_type
        self.acc_type: DataType = acc_type

    def __eq__(self, other):
        return isinstance(other, MmaConfig) and self.name == other.name

    def hidet_mma_config(self):
        from hidet.ir.primitives.cuda.mma import MmaConfig

        v_pos = self.name.find("v")
        under_pos = self.name.find("_", v_pos)
        hidet_config_name = self.name[:v_pos] + self.name[under_pos:]

        return getattr(MmaConfig, hidet_config_name)()

    @staticmethod
    def m16n8k16_f16_f16(vec_k: int = 1):
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
    def m16n8k16_f16_f32(vec_k: int = 1):
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
    def m16n8k16_bf16_f32(vec_k: int = 1):
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
    def m8n8k16_i8_i32(vec_k: int = 1):
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
    def all():
        config_list = []
        for vec_k in [1, 2, 3, 4]:
            config_list.append(MmaConfig.m16n8k16_f16_f32(vec_k))
            config_list.append(MmaConfig.m16n8k16_f16_f16(vec_k))
            config_list.append(MmaConfig.m16n8k16_bf16_f32(vec_k))
            config_list.append(MmaConfig.m8n8k16_i8_i32(vec_k))
        return {config.name: config for config in config_list}

    @staticmethod
    def from_name(name: str):
        return MmaConfig.all()[name]
