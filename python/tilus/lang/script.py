from __future__ import annotations

import typing
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Type, Union

from hidet.ir.dtypes import boolean
from hidet.ir.expr import Constant, Equal, Expr, LogicalAnd, Mod, Var, as_expr
from hidet.ir.primitives.cuda.vars import blockIdx, dim3, gridDim
from hidet.ir.tools import infer_type
from hidet.ir.type import DataType

from tilus.ir.builders import StmtBuilder
from tilus.ir.inst import InstructionError
from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout, global_repeat, global_strides
from tilus.ir.prog import Program
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.lang.modules.cuda import cuda
from tilus.lang.modules.utils import utils


class Attributes:
    """
    Attributes of the script program.

    Attributes
    ----------
    blocks: Optional[Sequence[Expr | int] | Expr | int]
        The number of blocks.
    warps: Optional[int]
        The number of warps, must between 1 and 32.
    """

    blocks: Optional[Sequence[Expr | int] | Expr | int] = None
    warps: Optional[int] = None

    def __setattr__(self, key, value):
        """Check the validity of the attribute value."""
        if key == "warps":
            if value is None:
                pass
            elif not isinstance(value, int):
                raise ValueError("The number of warps must be an integer")
            elif value <= 0:
                raise ValueError("The number of warps must be positive")
            elif value > 32:
                raise ValueError("The number of warps must be less than or equal to 32")
        elif key == "blocks":
            if value is None:
                pass
            elif not isinstance(value, (int, Expr)) and not isinstance(value, Sequence):
                raise ValueError("The number of blocks must be an integer or a sequence of integers")
            elif isinstance(value, Sequence):
                if not all(isinstance(v, (int, Expr)) for v in value):
                    raise ValueError("The number of blocks must be an integer or a sequence of integers")
        else:
            raise ValueError(f"Unknown attribute {key}")
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        ret = super().__getattribute__(key)
        if ret is None:
            raise ValueError("The function attribute {} must be set in the script program".format(key))
        return ret


class Script:
    # the compiled program will print the instruction output of the specified block
    debug_block: Optional[tuple[int, int, int]] = None

    # specify the schedule used for debugging, this will override any autotune space
    debug_schedule: Optional[dict[str, Any]] = None

    def __new__(cls, *args, **kwargs):
        from tilus.lang.instantiated_script import InstantiatedScriptCache

        return InstantiatedScriptCache.get(
            script_cls=cls,
            script_args=args,
            script_kwargs=kwargs,
        )

    def __init__(self) -> None:
        # builder used to append instructions
        from tilus.lang.transpiler import Transpiler

        self._builder: Optional[StmtBuilder] = None
        self._transpiler: Optional[Transpiler] = None

        # the following attributes should be set by the user in the kernel function
        self.attrs: Attributes = Attributes()
        self.blockIdx: dim3 = blockIdx
        self.gridDim: dim3 = gridDim

        # the following primitives could be used in the __init__ function to prepare the layouts
        self.cuda = cuda
        self.utils = utils

    def program(self) -> Program:
        """
        Get the traced program.

        The user defined script should satisfy:
        - 1) there is only one schedule .
        - 2) there is not const and tuning parameters in __call__.

        Returns
        -------
        ret: Program
            The traced program.
        """
        raise RuntimeError("This method should never be called. See InstantiatedScript.program instead.")

    def jit_instance(self, *args, **kwargs):
        """
        Instantiate the script program with the specified arguments and keyword arguments.

        Parameters
        ----------
        args:
            The positional arguments to the __call__ method.
        kwargs:
            The keyword arguments to the __call__ method.

        Returns
        -------
        ret: JitInstance
            The JIT instance for the script with given arguments.
        """
        raise RuntimeError("This method should never be called. See InstantiatedScript.jit_instance instead.")

    # the following functions should only be called in the __call__ function to construct the script program

    @staticmethod
    def range(
        start: Expr | int,
        end: Optional[Expr | int] = None,
        step: Optional[Expr | int] = None,
        /,
        *,
        unroll: Optional[Literal["all"] | int],
    ) -> Iterable[Var]:
        from tilus.lang.constructs.loops import range

        # the cast is to make the type checker happy
        return typing.cast(Iterable[Var], range(start, end, step, unroll=unroll))

    def assume(self, cond: Expr | bool) -> None:
        if isinstance(cond, bool):
            if not cond:
                raise InstructionError("The condition must be True")
            return
        if not isinstance(cond, Expr):
            raise InstructionError("The condition must be a boolean expression")

        # decompose the condition into conjuncture terms
        stack = [cond]
        terms: list[Expr] = []
        while stack:
            expr = stack.pop()
            if isinstance(expr, LogicalAnd):
                stack.append(expr.a)
                stack.append(expr.b)
            else:
                terms.append(expr)

        # analyze the conjunctures
        for term in terms:
            # a % c == 0
            if (
                isinstance(term, Equal)
                and isinstance(term.a, Mod)
                and isinstance(term.a.b, Constant)
                and isinstance(term.a.a, Var)
                and isinstance(term.b, Constant)
                and term.b.value == 0
            ):
                a = term.a.a
                if a not in self._transpiler.func_params:
                    raise InstructionError(
                        "We only allow to specify the divisibility of kernel parameter, got {}".format(a.name)
                    )
                self._transpiler.var2divisibility[a] = int(term.a.b.value)  # type: ignore[arg-type]
            else:
                raise InstructionError("Can not recognize the condition in assume: {}".format(term))

    def register_tensor(
        self,
        *,
        dtype: DataType,
        shape: Optional[Sequence[int]] = None,
        layout: Optional[RegisterLayout] = None,
        f_init: Optional[Callable[[Sequence[Var]], Expr]] = None,
        init: Optional[Expr | int | float] = None,
    ) -> RegisterTensor:
        if f_init is not None and init is not None:
            raise ValueError("Cannot specify both f_init and init")
        elif f_init is None and init is not None:

            def f_init(_):
                return dtype.constant(init)

        return self._builder.allocate_register(dtype=dtype, shape=shape, layout=layout, f_init=f_init)

    def global_tensor(
        self,
        dtype: DataType,
        shape: Optional[Sequence[int]] = None,
        layout: Optional[GlobalLayout] = None,
        *,
        requires_clean: bool,
    ) -> GlobalTensor:
        return self._builder.allocate_global(
            dtype=dtype,
            shape=shape,
            layout=layout,
            requires_clean=requires_clean,
        )

    def shared_tensor(
        self,
        *,
        dtype: DataType,
        shape: Optional[Sequence[int]] = None,
        layout: Optional[SharedLayout] = None,
    ) -> SharedTensor:
        return self._builder.allocate_shared(dtype=dtype, shape=shape, layout=layout)

    def global_view(
        self,
        ptr: Expr,
        *,
        dtype: DataType,
        shape: Optional[Sequence[Expr | int]] = None,
        strides: Optional[Sequence[Expr | int]] = None,
        layout: Optional[GlobalLayout] = None,
    ) -> GlobalTensor:
        if layout is not None:
            assert shape is None and strides is None, "Cannot specify both layout and shape/strides"
            layout = layout
        else:
            assert shape is not None, "Must specify shape when layout is not provided"
            if strides is None:
                # assume compact row-major layout
                layout = global_repeat(*shape)
            else:
                assert len(shape) == len(strides), "Shape and strides must have the same length"
                layout = global_strides(shape, strides)

        return self._builder.global_view(ptr=ptr, dtype=dtype, layout=layout)

    def load_global(
        self,
        x: GlobalTensor,
        /,
        *,
        offsets: Sequence[Expr | int],
        shape: Optional[Sequence[int]] = None,
        layout: Optional[RegisterLayout] = None,
        slice_dims: Optional[Sequence[int]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if len(offsets) != len(x.shape):
            raise InstructionError(
                "The number of offsets must be equal to the number of dimensions of the global tensor"
            )
        return self._builder.load_global(
            x=x, offsets=offsets, slice_dims=slice_dims, shape=shape, layout=layout, output=out
        )

    def store_global(
        self,
        dst: GlobalTensor,
        x: RegisterTensor,
        *,
        offsets: Sequence[Expr | int],
        slice_dims: Optional[Sequence[int]] = None,
    ) -> None:
        if slice_dims is not None and len(slice_dims) != len(x.shape):
            raise InstructionError(
                "The number of slice dimensions must be equal to the number of dimensions of the "
                f"register tensor: {len(slice_dims)} vs {len(x.shape)}"
            )
        return self._builder.store_global(dst=dst, src=x, offsets=offsets, dims=slice_dims)

    def load_shared(
        self,
        src: SharedTensor,
        *,
        layout: Optional[RegisterLayout] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.load_shared(src=src, layout=layout, output=out)

    def store_shared(
        self,
        dst: SharedTensor,
        src: RegisterTensor,
        *,
        offsets: Optional[Sequence[int]] = None,
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        if dst.dtype != src.dtype:
            raise InstructionError(
                "Cannot store shared tensor {}{} from register tensor {}{}: dtype mismatch".format(
                    dst.dtype.name, list(dst.shape), src.dtype.name, list(src.shape)
                )
            )
        if offsets is not None:
            assert len(offsets) == len(dst.shape)
            if dims is None:
                assert len(src.shape) == len(dst.shape)
                dims = list(range(len(src.shape)))
            dst = self._builder.shared_slice(dst, offsets=offsets, slice_dims=dims, slice_shape=src.shape)
        self._builder.store_shared(dst=dst, src=src)

    def free_shared(self, tensor: SharedTensor) -> None:
        self._builder.free_shared(tensor)

    def copy_async(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        weak_mask: bool = False,
    ) -> None:
        self._builder.copy_async(dst=dst, src=src, offsets=offsets, dims=dims, evict=evict, weak_mask=weak_mask)

    def copy_async_wait_all(self):
        self._builder.copy_async_wait_all()

    def copy_async_commit_group(self):
        self._builder.copy_async_commit_group()

    def copy_async_wait_group(self, n: Union[Expr, int]) -> None:
        self._builder.copy_async_wait_group(n)

    def mma_dot(
        self,
        a: RegisterTensor,
        b: RegisterTensor,
        c: Optional[RegisterTensor] = None,
        /,
        *,
        acc_dtype: Optional[DataType] = None,
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if c is None:
            if acc_dtype is None:
                raise InstructionError('mma_dot requires either "c" or "acc_dtype" to be specified')
            m, n = a.shape[-2], b.shape[-1]
            c = self._builder.allocate_register(
                dtype=acc_dtype,
                shape=[m, n],
                f_init=lambda _: acc_dtype.constant(0),
            )
        else:
            if acc_dtype is not None and acc_dtype != c.dtype:
                raise InstructionError(
                    "The dtype of the accumulator tensor 'c' must match the specified 'acc_dtype' if provided"
                )
        return self._builder.mma_dot(
            a,
            b,
            c,
            output=output,
        )

    def cast(self, x: RegisterTensor, dtype: DataType) -> RegisterTensor:
        return self._builder.cast(x=x, dtype=dtype)

    def view(
        self,
        x: RegisterTensor,
        *,
        layout: Optional[RegisterLayout] = None,
        dtype: Optional[DataType] = None,
        local_offset: Union[Expr, int] = 0,
    ) -> RegisterTensor:
        return self._builder.view(x=x, layout=layout, dtype=dtype, local_offset=local_offset)

    def squeeze(
        self,
        x: RegisterTensor,
        *,
        dim: int,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.squeeze(x, dim=dim, out=out)

    def unsqueeze(
        self,
        x: RegisterTensor,
        *,
        dim: int | Sequence[int],
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.unsqueeze(x, dim=dim, out=out)

    def transpose(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.transpose(x, out=out)

    def abs(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.abs(x, out=out)

    def exp(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.exp(x, out=out)

    def round(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.round(x, out=out)

    def clip(
        self,
        x: RegisterTensor,
        min: Expr | int | float,
        max: Expr | int | float,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.clip(x=x, min=min, max=max, out=out)

    def repeat(
        self,
        x: RegisterTensor,
        repeats: Sequence[int],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.repeat(
            x=x,
            repeats=repeats,
            out=out,
        )

    def repeat_interleave(
        self,
        x: RegisterTensor,
        repeats: Sequence[int],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.repeat_interleave(
            x=x,
            repeats=repeats,
            out=out,
        )

    def load_global_generic(
        self,
        *,
        dtype: DataType,
        layout: RegisterLayout,
        ptr: Var,
        f_offset: Callable[..., Expr | int],
        f_mask: Optional[Callable[..., Expr | int | bool]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.load_global_generic(
            dtype=dtype,
            layout=layout,
            ptr=ptr,
            f_offset=lambda args: f_offset(*args),
            f_mask=lambda args: f_mask(*args) if f_mask is not None else None,
            out=out,
        )

    def _reduce(
        self,
        x: RegisterTensor,
        *,
        dim: int,
        keepdim: bool,
        op: str,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.reduce(x, dim=dim, keepdim=keepdim, op=op, out=out)

    def sum(
        self, x: RegisterTensor, *, dim: int, keepdim: bool = False, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self._reduce(x, dim=dim, keepdim=keepdim, op="sum", out=out)

    def max(
        self, x: RegisterTensor, *, dim: int, keepdim: bool = False, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self._reduce(x, dim=dim, keepdim=keepdim, op="max", out=out)

    def min(
        self, x: RegisterTensor, *, dim: int, keepdim: bool = False, out: Optional[RegisterTensor] = None
    ) -> RegisterTensor:
        return self._reduce(x, dim=dim, keepdim=keepdim, op="min", out=out)

    def store_global_generic(
        self,
        x: RegisterTensor,
        /,
        *,
        ptr: Var,
        f_offset: Callable[..., Expr | int],
        f_mask: Optional[Callable[..., Expr | int | bool]] = None,
    ) -> None:
        self._builder.store_global_generic(
            x=x,
            ptr=ptr,
            f_offset=lambda args: f_offset(*args),
            f_mask=lambda args: f_mask(*args) if f_mask is not None else None,
        )

    def add(self, lhs: RegisterTensor, rhs: RegisterTensor, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._builder.add(lhs, rhs, out=out)

    def maximum(self, lhs: RegisterTensor, rhs: RegisterTensor, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        return self._builder.maximum(lhs, rhs, out=out)

    def where(
        self,
        condition: RegisterTensor,
        x: RegisterTensor | Expr | int | float,
        y: RegisterTensor | Expr | int | float,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if not isinstance(condition, RegisterTensor):
            cond_expr = as_expr(condition)
            condition = self._builder.allocate_register(dtype=boolean, shape=(), f_init=lambda _: cond_expr)
        if not isinstance(x, RegisterTensor):
            x_expr = as_expr(x)
            x = self._builder.allocate_register(dtype=infer_type(x), shape=(), f_init=lambda _: x_expr)
        if not isinstance(y, RegisterTensor):
            y_expr = as_expr(y)
            y = self._builder.allocate_register(dtype=infer_type(y), shape=(), f_init=lambda _: y_expr)
        if condition.dtype != boolean:
            raise InstructionError("Condition must be a boolean tensor, got {}".format(condition.dtype))
        if x.dtype != y.dtype:
            raise InstructionError("The types of x and y must match, got {} and {}".format(x.dtype, y.dtype))
        return self._builder.where(condition, x, y, out=out)

    def lock_semaphore(self, semaphore: Expr, value: Expr | int) -> None:
        self._builder.lock_semaphore(semaphore, value)

    def release_semaphore(self, semaphore: Expr, value: Expr | int) -> None:
        self._builder.release_semaphore(semaphore, value)

    def sync(self) -> None:
        self._builder.syncthreads()

    def print_tensor(self, msg: str, tensor: Tensor, fmt: Optional[str] = None) -> None:
        self._builder.print_tensor(msg=msg, tensor=tensor, fmt=fmt)

    def printf(self, fstring: str, *args: Expr | int | float) -> None:
        self._builder.printf(fstring, *args)

    @staticmethod
    def static_assert(cond: bool | Expr, msg: str) -> None:
        if not isinstance(cond, Constant) and not isinstance(cond, bool):
            raise ValueError("Static assert condition must be a constant")
        if not cond:
            raise AssertionError(msg)


def autotune(arg_names: str, arg_values: Sequence[Any]) -> Callable[[Type[Script]], Any]:
    def decorator(script_cls):
        if not hasattr(script_cls, "_autotune_space"):
            script_cls._autotune_space = {}
        space = getattr(script_cls, "_autotune_space")
        names = [name.strip() for name in arg_names.split(",")]
        if any(name in space for name in names):
            common_names = set(names) & set(space.keys())
            raise RuntimeError("Duplicated specification for parameters: {}".format(common_names))
        space[arg_names] = arg_values
        setattr(script_cls, "_autotune_space", space)

        # return functools.wraps(wrapped=script_cls, assigned=arg_names)(script_cls)
        return script_cls

    return decorator
