from __future__ import annotations
from typing import List, Union, Optional, Callable, Dict, Any, Tuple, Sequence, Mapping
from functools import lru_cache
from hidet.ir.expr import Var, Expr
from hidet.ir.type import DataType, BaseType, PointerType
from hidet.ir.dtypes import bf16, f16, f32, i32, i8, boolean
from hidet.utils.py import is_power_of_two, prod
from tilus.extensions.hidet.ir.expr import index_vars, convert_to_expr
from tilus.ir.layouts import Layout, spatial, repeat, column_repeat, column_spatial
from tilus.ir.value import Value, RegisterValue, SharedValue, SharedLayout


class Instruction:
    def __init__(self, output: Optional[Value], inputs: List[Value], attrs: Dict[str, Any]):
        self._output: Optional[Value] = output
        self._inputs: List[Value] = inputs
        self._attrs: Dict[str, Any] = attrs

    def __str__(self):
        items = []
        if self._output is not None:
            items.append(str(self._output))
        items.extend([str(t) for t in self.inputs])
        items.extend([f"{k}={v}" for k, v in self.attrs.items()])
        return "{}({})".format(self.__class__.__name__, ",\n".join(items))

    @property
    def output(self) -> Value:
        if self._output is None:
            raise ValueError("output is not set")
        return self._output

    @property
    def shared_output(self) -> SharedValue:
        assert isinstance(self._output, SharedValue)
        return self._output

    @property
    def register_output(self) -> RegisterValue:
        assert isinstance(self._output, RegisterValue)
        return self._output

    @property
    def inputs(self) -> Sequence[Value]:
        return self._inputs

    @property
    def attrs(self) -> Mapping[str, Any]:
        return self._attrs

    def has_output(self) -> bool:
        return self._output is not None

    def recreate(self, updated_output: Optional[Value], updated_inputs: List[Value], updated_attrs: Dict[str, Any]):
        # by default, all subclasses of Instruction will have __init__ accepts ([output, ]*inputs, **attrs)
        # if the subclass has different signature, it should override this method so that we can recreate the instance
        # with the updated values and attrs
        values = []
        if updated_output:
            values.append(updated_output)
        values.extend(updated_inputs)
        return type(self)(*values, **updated_attrs)  # type: ignore

    def request_shared_workspace(self) -> int:
        return 0


class AssignInst(Instruction):
    def __init__(self, output: Value, x: Value):
        super().__init__(output, inputs=[x], attrs={})

    @staticmethod
    def create(output: Value, x: Value):
        return AssignInst(output, x)


class AllocateInst(Instruction):
    def __init__(self, output: RegisterValue, axes: Optional[List[Var]], init: Optional[Expr]):
        super().__init__(output, inputs=[], attrs={"axes": axes, "init": init})
        self.axes: Optional[List[Var]] = axes
        self.init: Optional[Expr] = init

    @staticmethod
    def create(dtype: DataType, layout: Layout, f_init: Optional[Callable[[List[Var]], Expr]] = None) -> AllocateInst:
        out = RegisterValue(dtype, layout)
        if f_init is not None:
            axes = index_vars(num_vars=len(layout.shape))
            init = f_init(axes)
        else:
            axes = None
            init = None
        return AllocateInst(out, axes, init)


class AllocateScalarInst(Instruction):
    def __init__(self, var: Var, init: Optional[Expr]):
        super().__init__(output=None, inputs=[], attrs={"var": var, "init": init})
        self.var: Var = var
        self.init: Optional[Expr] = init

    @staticmethod
    def create(hint: str, scalar_type: Union[DataType, PointerType], init: Optional[Expr] = None):
        var = Var(hint=hint, type=scalar_type)
        return AllocateScalarInst(var, init)


class AssignScalarInst(Instruction):
    def __init__(self, var: Var, scalar_expr: Expr):
        super().__init__(output=None, inputs=[], attrs={"var": var, "scalar_expr": scalar_expr})
        self.var: Var = var
        self.scalar_expr: Expr = scalar_expr

    @staticmethod
    def create(var: Var, scalar_expr: Expr):
        return AssignScalarInst(var, scalar_expr)


class LoadGlobalInst(Instruction):
    def __init__(self, output: RegisterValue, ptr: Var, axes: List[Var], offset: Expr, mask: Optional[Expr]):
        super().__init__(output=output, inputs=[], attrs={"ptr": ptr, "axes": axes, "offset": offset, "mask": mask})
        self.ptr: Var = ptr
        self.axes: List[Var] = axes
        self.offset: Expr = offset
        self.mask: Optional[Expr] = mask

        assert isinstance(offset, Expr) and (mask is None or isinstance(mask, Expr))

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
            out = RegisterValue(dtype, layout)
        axes = index_vars(num_vars=len(layout.shape))
        offset = convert_to_expr(f_offset(axes))
        mask = convert_to_expr(f_mask(axes)) if f_mask is not None else None
        return LoadGlobalInst(out, ptr, axes, offset, mask)


class StoreGlobalInst(Instruction):
    def __init__(self, x: RegisterValue, ptr: Var, axes: List[Var], offset: Expr, mask: Optional[Expr]):
        super().__init__(output=None, inputs=[x], attrs={"ptr": ptr, "axes": axes, "offset": offset, "mask": mask})
        self.ptr: Var = ptr
        self.axes: List[Var] = axes
        self.offset: Expr = offset
        self.mask: Optional[Expr] = mask

        assert ptr is not None

    @staticmethod
    def create(
        x: RegisterValue,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr | int],
        f_mask: Optional[Callable[[List[Var]], Expr | int | bool]] = None,
    ) -> StoreGlobalInst:
        axes = index_vars(num_vars=len(x.layout.shape))
        offset = convert_to_expr(f_offset(axes))
        mask = convert_to_expr(f_mask(axes)) if f_mask is not None else None
        return StoreGlobalInst(x, ptr, axes, offset, mask)


class CastInst(Instruction):
    def __init__(
        self,
        output: RegisterValue,
        x: RegisterValue,
        interleave_width: Optional[int],
        interleave_stride: Optional[int],
        ignore_int4b_xor: bool = False,
    ):
        super().__init__(
            output=output,
            inputs=[x],
            attrs={
                "interleave_width": interleave_width,
                "interleave_stride": interleave_stride,
                "ignore_int4b_xor": ignore_int4b_xor,
            },
        )
        self.interleave_width: Optional[int] = interleave_width
        self.interleave_stride: Optional[int] = interleave_stride
        self.ignore_int4b_xor: bool = ignore_int4b_xor  # todo: remove this hack

        assert not ((interleave_width is not None) ^ (interleave_stride is not None))

    @staticmethod
    def create(
        dtype: DataType, x: RegisterValue, interleave_width=None, interleave_stride=None, ignore_int4b_xor=False
    ) -> CastInst:
        return CastInst(RegisterValue(dtype, x.layout), x, interleave_width, interleave_stride, ignore_int4b_xor)


class ElementwiseUnaryInst(Instruction):
    VALID_OPS = ["relu", "clip"]

    def __init__(self, output: RegisterValue, x: RegisterValue, op: str, other_kwargs):
        super().__init__(output=output, inputs=[x], attrs={"op": op, **other_kwargs})
        self.op: str = op

        assert op in self.VALID_OPS


class ElementwiseBinaryInst(Instruction):
    VALID_OPS = ["+", "-", "*", "/", "%"]

    def __init__(self, output: RegisterValue, x: RegisterValue, y: RegisterValue, op: str):
        super().__init__(output=output, inputs=[x, y], attrs={"op": op})
        self.op: str = op

    @staticmethod
    def create(x: RegisterValue, y: RegisterValue, op: str, *, output: Optional[RegisterValue]):
        assert op in ElementwiseBinaryInst.VALID_OPS
        if output is None:
            output = RegisterValue(x.dtype, x.layout)
        return ElementwiseBinaryInst(output, x, y, op)


class BroadcastElementwiseBinaryInst(Instruction):
    VALID_OPS = ["+", "-", "*", "/", "%"]

    def __init__(self, output: RegisterValue, r: RegisterValue, s: Expr, op: str, tensor_left: bool):
        super().__init__(output=output, inputs=[r], attrs={"s": s, "op": op, "tensor_left": tensor_left})
        self.op: str = op
        self.s: Expr = s
        self.tensor_left: bool = tensor_left

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
            output = RegisterValue(out_dtype, r.layout)
        return BroadcastElementwiseBinaryInst(output, r, s, op, tensor_left=tensor_left)


class MmaDotInst(Instruction):
    def __init__(
        self,
        d: RegisterValue,
        a: RegisterValue,
        b: RegisterValue,
        c: RegisterValue,
        mma_inst: str,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
    ):
        super().__init__(
            output=d,
            inputs=[a, b, c],
            attrs={"mma_inst": mma_inst, "warp_spatial": warp_spatial, "warp_repeat": warp_repeat},
        )
        self.mma_inst: str = mma_inst
        self.warp_spatial: Tuple[int, int, int] = warp_spatial
        self.warp_repeat: Tuple[int, int, int] = warp_repeat

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
            output = RegisterValue(c.dtype, c.layout)
        return MmaDotInst(
            d=output, a=a, b=b, c=c, mma_inst=mma_inst, warp_spatial=warp_spatial, warp_repeat=warp_repeat
        )


class SimtDotInst(Instruction):
    def __init__(
        self,
        d: RegisterValue,
        a: RegisterValue,
        b: RegisterValue,
        c: RegisterValue,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        thread_spatial: Tuple[int, int],
        thread_repeat: Tuple[int, int],
    ):
        super().__init__(
            output=d,
            inputs=[a, b, c],
            attrs={
                "warp_spatial": warp_spatial,
                "warp_repeat": warp_repeat,
                "thread_spatial": thread_spatial,
                "thread_repeat": thread_repeat,
            },
        )
        self.warp_spatial: Tuple[int, int, int] = warp_spatial
        self.warp_repeat: Tuple[int, int, int] = warp_repeat
        self.thread_spatial: Tuple[int, int] = thread_spatial
        self.thread_repeat: Tuple[int, int] = thread_repeat


class FormatPrintInst(Instruction):
    def __init__(self, cond: Expr, fstring: str, expressions: Sequence[Expr] = tuple()):
        super().__init__(
            output=None, inputs=[], attrs={"cond": cond, "fstring": fstring, "expressions": list(expressions)}
        )
        self.cond: Expr = cond
        self.fstring: str = fstring
        self.expressions: List[Expr] = list(expressions)

        assert cond is not None


class PrintValueInst(Instruction):
    def __init__(self, x: Value, cond: Expr, msg: str, fmt: Optional[str] = None):
        super().__init__(output=None, inputs=[x], attrs={"cond": cond, "msg": msg, "fmt": fmt})
        self.cond: Expr = cond
        self.msg: str = msg
        self.fmt: Optional[str] = fmt


class ShuffleBaseInst(Instruction):
    def __init__(self, output: Value, x: Value, mask: int, delta: int, width: int):
        super().__init__(output=output, inputs=[x], attrs={"mask": mask, "delta": delta, "width": width})
        self.mask: int = mask
        self.delta: int = delta
        self.width: int = width

        warp_size = 32
        self.dtype: DataType = output.as_register_value().dtype
        self.layout: Layout = output.as_register_value().layout
        self.num_warps: int = self.layout.num_workers // 32
        self.num_groups: int = max([i // width for i in range(self.num_warps) if mask & (1 << i)]) + 1
        self.smem_shape: Tuple[int, int, int] = (self.num_groups, width - delta, warp_size * self.layout.local_size)

        assert all(is_power_of_two(v) for v in [delta, width, self.num_warps])

    def request_shared_workspace(self) -> int:
        return self.dtype.nbytes * prod(self.smem_shape)


class ShuffleDownInst(ShuffleBaseInst):
    pass


class ShuffleUpInst(ShuffleBaseInst):
    pass


# class ShuffleInst(Instruction):
#     def __init__(self, output: Value, x: Value, mask: int, src_lane: int, width: int):
#         super().__init__(output=output, inputs=[x], attrs={'mask': mask, 'src_lane': src_lane, 'width': width})
#         self.mask: int = mask
#         self.src_lane: int = src_lane
#         self.width: int = width


class ViewInst(Instruction):
    def __init__(self, output: RegisterValue, x: RegisterValue, local_offset: Expr):
        super().__init__(output=output, inputs=[x], attrs={"local_offset": local_offset})
        self.local_offset: Expr = local_offset

        assert output.layout.local_size * output.dtype.nbits <= x.layout.local_size * x.dtype.nbits
        assert output.layout.num_workers <= x.layout.num_workers

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
        output = RegisterValue(dtype=dtype, layout=layout)
        return ViewInst(output=output, x=x, local_offset=i32(local_offset))


class AllocateSharedInst(Instruction):
    def __init__(self, output: SharedValue, axes: Optional[List[Var]], init: Optional[Expr]):
        super().__init__(output=output, inputs=[], attrs={"axes": axes, "init": init})
        self.axes: Optional[List[Var]] = axes
        self.init: Optional[Expr] = init

    @staticmethod
    def create(
        dtype: DataType, shared_layout: SharedLayout, f_init: Optional[Callable[[List[Var]], Expr]] = None
    ) -> AllocateSharedInst:
        out = SharedValue(dtype=dtype, layout=shared_layout)
        if f_init:
            axes = index_vars(num_vars=len(shared_layout.shape))
            init = f_init(axes)
        else:
            axes = None
            init = None
        return AllocateSharedInst(output=out, axes=axes, init=init)


class AllocateGlobalInst(Instruction):
    def __init__(self, var: Var, nbytes: Expr, require_clean: bool = False):
        super().__init__(output=None, inputs=[], attrs={"var": var, "nbytes": nbytes, "require_clean": require_clean})
        self.var: Var = var
        self.nbytes: Expr = nbytes
        self.require_clean: bool = require_clean

    @staticmethod
    def create(hint: str, scalar_type: BaseType, nbytes: Union[Expr, int], require_clean: bool = False):
        return AllocateGlobalInst(
            var=Var(hint, scalar_type),
            nbytes=nbytes if isinstance(nbytes, Expr) else i32(nbytes),
            require_clean=require_clean,
        )


class FreeSharedInst(Instruction):
    def __init__(self, shared_value: SharedValue):
        super().__init__(output=None, inputs=[shared_value], attrs={})

    @staticmethod
    def create(shared_value: SharedValue):
        return FreeSharedInst(shared_value)


class ViewSharedInst(Instruction):
    def __init__(self, out: SharedValue, x: SharedValue, indices: List[Expr], dtype: DataType, layout: SharedLayout):
        super().__init__(output=out, inputs=[x], attrs={"indices": indices, "layout": layout, "dtype": dtype})
        self.indices: List[Expr] = indices
        self.dtype: DataType = dtype
        self.layout: SharedLayout = layout

    @staticmethod
    def create(x: SharedValue, indices: List[Expr], layout: SharedLayout, dtype: Optional[DataType] = None):
        if dtype is None:
            dtype = x.dtype
        out = SharedValue(dtype=dtype, layout=layout)
        return ViewSharedInst(out=out, x=x, indices=indices, dtype=dtype, layout=layout)


class CopyAsyncInst(Instruction):
    def __init__(
        self,
        dst: SharedValue,
        ptr: Var,
        axes: List[Var],
        offset: Expr,
        mask: Optional[Expr] = None,
        evict: Optional[str] = None,
    ):
        super().__init__(
            output=None, inputs=[dst], attrs={"ptr": ptr, "axes": axes, "offset": offset, "mask": mask, "evict": evict}
        )
        self.ptr: Var = ptr
        self.axes: List[Var] = axes
        self.offset: Expr = offset
        self.mask: Optional[Expr] = mask
        self.evict: Optional[str] = evict

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
        return CopyAsyncInst(dst, ptr=ptr, axes=axes, offset=offset, mask=mask, evict=evict)


class CopyAsyncCommitGroupInst(Instruction):
    def __init__(self):
        super().__init__(output=None, inputs=[], attrs={})


class CopyAsyncWaitGroupInst(Instruction):
    def __init__(self, n: Expr):
        super().__init__(output=None, inputs=[], attrs={"n": n})
        self.n: Expr = n


class CopyAsyncWaitAllInst(Instruction):
    def __init__(self):
        super().__init__(output=None, inputs=[], attrs={})


class SyncThreadsInst(Instruction):
    def __init__(self):
        super().__init__(output=None, inputs=[], attrs={})


class SyncReduceThreadsInst(Instruction):
    AND = "and"
    OR = "or"

    def __init__(self, reduce_op: str, var: Var, reduce_value: Expr):
        super().__init__(
            output=None, inputs=[], attrs={"reduce_op": reduce_op, "var": var, "reduce_value": reduce_value}
        )
        self.var: Var = var
        self.reduce_op: str = reduce_op
        self.reduce_value: Expr = reduce_value

        assert reduce_op in [SyncReduceThreadsInst.AND, SyncReduceThreadsInst.OR]

    @staticmethod
    def create(reduce_op: str, var_hint: str, reduce_value: Expr):
        var = Var(var_hint, type=boolean)
        return SyncReduceThreadsInst(reduce_op, var, reduce_value)


class LoadScalarInst(Instruction):
    def __init__(self, var: Var, ptr: Expr, sync: str = "weak"):
        super().__init__(output=None, inputs=[], attrs={"var": var, "ptr": ptr, "sync": sync})
        self.var: Var = var
        self.ptr: Expr = ptr
        self.sync: str = sync
        assert sync in ["weak", "acquire"]

    @staticmethod
    def create(ptr: Expr, sync: str = "weak"):
        from hidet.ir.tools import infer_type

        ptr_type = infer_type(ptr)
        assert isinstance(ptr_type, PointerType)
        var_type = ptr_type.base_type
        assert isinstance(var_type, DataType)
        var = Var("loaded_{}".format(var_type.name), var_type)
        return LoadScalarInst(var=var, ptr=ptr, sync=sync)


class StoreScalarInst(Instruction):
    def __init__(self, ptr: Expr, value: Expr, sync: str = "weak"):
        super().__init__(output=None, inputs=[], attrs={"ptr": ptr, "value": value, "sync": sync})
        self.ptr: Expr = ptr
        self.value: Expr = value
        self.sync: str = sync

        assert sync in ["weak", "release"]

    @staticmethod
    def create(ptr: Expr, value: Expr, sync: str = "weak"):
        return StoreScalarInst(ptr=ptr, value=value, sync=sync)


class AtomicScalarInst(Instruction):
    """
    *ptr = op(*ptr, value) in atomic way
    """

    def __init__(self, ptr: Expr, op: str, value: Expr):
        super().__init__(output=None, inputs=[], attrs={"ptr": ptr, "op": op, "value": value})
        self.ptr: Expr = ptr
        self.op: str = op
        self.value: Expr = value

        assert op in ["add", "sub", "min", "max"]

    @staticmethod
    def create(ptr: Expr, op: str, value: Expr):
        return AtomicScalarInst(ptr=ptr, op=op, value=value)


class LoadMatrixInst(Instruction):
    ldmatrix_configs = [
        # (dtype bytes, trans, ldmatrix_layout)
        (1, False, spatial(8, 4).repeat(1, 4)),
        # (1, True, column_spatial(4, 8).repeat(4, 1)), # ldmatrix does not support this case
        (2, False, spatial(8, 4).repeat(1, 2)),
        (2, True, column_spatial(4, 8).repeat(2, 1)),
    ]

    def __init__(self, output: RegisterValue, src: SharedValue, offsets: List[Expr]):
        super().__init__(output=output, inputs=[src], attrs={"offsets": offsets})
        self.offsets: List[Expr] = offsets

    @staticmethod
    def create(
        src: SharedValue, register_layout: Layout, offsets: List[Expr], output: Optional[RegisterValue] = None
    ) -> LoadMatrixInst:
        if output is None:
            output = RegisterValue(dtype=src.dtype, layout=register_layout)
        else:
            assert output.dtype == src.dtype and output.layout.quick_equal(register_layout)
        return LoadMatrixInst(output=output, src=src, offsets=offsets)


class LoadSharedInst(Instruction):
    def __init__(self, output: RegisterValue, src: SharedValue, offsets: List[Expr]):
        super().__init__(output=output, inputs=[src], attrs={"offsets": offsets})
        self.offsets: List[Expr] = offsets

        assert len(src.shape) == len(output.shape) == len(offsets)

    @staticmethod
    def create(
        src: SharedValue, register_layout: Layout, offsets: List[Expr], *, output: Optional[RegisterValue] = None
    ) -> LoadSharedInst:
        if output is None:
            output = RegisterValue(dtype=src.dtype, layout=register_layout)
        else:
            assert output.dtype == src.dtype and output.layout.quick_equal(register_layout)
        return LoadSharedInst(output=output, src=src, offsets=offsets)


class StoreSharedInst(Instruction):
    def __init__(self, dst: SharedValue, src: RegisterValue, offsets: List[Expr]):
        super().__init__(output=None, inputs=[dst, src], attrs={"offsets": offsets})
        self.offsets: List[Expr] = offsets

        assert len(src.shape) == len(dst.shape) == len(offsets)

    @staticmethod
    def create(dst: SharedValue, src: RegisterValue, offsets: Optional[List[Expr]] = None) -> StoreSharedInst:
        if offsets is None:
            offsets = [i32.zero for _ in range(len(dst.shape))]
        return StoreSharedInst(dst, src, offsets)


class ExitInst(Instruction):
    def __init__(self):
        super().__init__(output=None, inputs=[], attrs={})

    @staticmethod
    def create() -> ExitInst:
        return ExitInst()


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
