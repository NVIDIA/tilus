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
    """Attributes of the script program."""

    _blocks: Optional[Sequence[Expr | int] | Expr | int] = None
    _warps: Optional[int] = None

    @property
    def blocks(self) -> Sequence[Expr | int] | Expr | int | None:
        """The number of blocks."""
        return self._blocks

    @blocks.setter
    def blocks(self, value: Sequence[Expr | int] | Expr | int) -> None:
        self._blocks = value

    @property
    def warps(self) -> Optional[int]:
        """The number of warps."""
        return self._warps

    @warps.setter
    def warps(self, value: int) -> None:
        if value is None:
            self._warps = None
        elif not isinstance(value, int):
            raise ValueError("The number of warps must be an integer")
        elif value <= 0:
            raise ValueError("The number of warps must be positive")
        elif value > 32:
            raise ValueError("The number of warps must be less than or equal to 32")
        else:
            self._warps = value


class Script:
    """A script is a user-defined kernel function that can be compiled and executed on the GPU."""

    # the compiled program will print the instruction output of the specified block
    debug_block: Optional[tuple[int, int, int]] = None

    # specify the schedule used for debugging. it will override any autotune space
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

        self._attrs: Attributes = Attributes()

        # the following primitives could be used in the __init__ function to prepare the layouts
        self.cuda = cuda
        self.utils = utils

    def __call__(self, *args, **kwargs):
        raise RuntimeError("This method should never be called.")

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

    def jit_instance_for(self, *args: object, **kwargs: object) -> Any:
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

    # the following properties should only be access in the __call__ function
    @property
    def attrs(self) -> Attributes:
        """Kernel attributes like number of blocks and warps.

        See :py:class:`Attributes <tilus.lang.Attributes>` for more details.
        """
        return self._attrs

    @property
    def blockIdx(self) -> dim3:
        """Get the block index of the current thread block."""
        return blockIdx

    @property
    def gridDim(self) -> dim3:
        """Get the grid dimension of the kernel."""
        return gridDim

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
        shape: Sequence[int],
        layout: Optional[RegisterLayout] = None,
        init: Optional[Callable[[Sequence[Var]], Expr | int | float | bool] | Expr | int | float] = None,
    ) -> RegisterTensor:
        """Create a register tensor.

        This instruction allocates a register tensor with the specified data type, shape, (optional) layout, and
        (optional) initialization value.

        When `layout` is not provided, the layout of the register tensor will be automatically inferred based on the
        operations performed on it.

        If `init` is not provided, the register tensor will be uninitialized. If `init` is provided, it can be a
        scalar value (e.g., `int`, `float`, `bool`, or a scalar expression) that will be used to initialize all elements
        of the register tensor. It can also be a callable function that takes a sequence of index variables (i, j, ...)
        and returns a scalar expression based on these indices. The element at index (i, j, ...) will be initialized
        with the value returned by this function. When the data type of the value is not identical to the data type of
        the register tensor, the value will be cast to the data type of the register tensor.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[int]
            The shape of the tensor.
        layout: RegisterLayout, optional
            The layout of the tensor. If not provided, the layout will be inferred based on the operations performed on it.
        init: Callable[[Sequence[Var]], Expr | int | float | bool] | Expr | int | float, optional
            The initialization value or function to initialize the tensor elements.

        Returns
        -------
        tensor: RegisterTensor
            The allocated register tensor.
        """
        if init is None:
            f_init = None
        elif isinstance(init, (float, int, bool, Expr)):
            f_init = lambda _: dtype.constant(init)  # noqa: E731
        elif callable(init):
            f_init = init
        else:
            raise ValueError("init must be a callable, int, float, bool, or Expr, got {}".format(type(init)))

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
        """Allocate a shared tensor.

        This instruction allocates a shared tensor with the specified data type, shape, and (optional) layout.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[int]
            The shape of the tensor.
        layout: SharedLayout, optional
            The layout of the tensor. If not provided, the layout will be inferred based on the operations performed
            on it.

        Returns
        -------
        ret: SharedTensor
            The allocated shared tensor.
        """
        return self._builder.allocate_shared(dtype=dtype, shape=shape, layout=layout)

    def global_view(
        self,
        ptr: Expr,
        *,
        dtype: DataType,
        shape: Sequence[Expr | int],
        strides: Optional[Sequence[Expr | int]] = None,
        layout: Optional[GlobalLayout] = None,
    ) -> GlobalTensor:
        """Create a global tensor view.

        There are three ways to specify the layout:

        - `layout`: If provided, it overrides the shape and strides parameters.
        - `shape`: If provided, it defines the shape of the tensor and assume a compact row-major strides.
        - `shape` and `strides`: If provided, they define the shape and strides of the tensor.

        Parameters
        ----------
        ptr: Expr
            The pointer to the global memory, which should be a pointer expression to the first element of the tensor.
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[Expr | int]
            The shape of the tensor.
        strides: Sequence[Expr | int], optional
            The strides of the tensor. If not provided, it is assumed to be compact row-major layout.
        layout: GlobalLayout, optional
            The layout of the tensor. If provided, it overrides the shape and strides parameters.

        Returns
        -------
        ret: GlobalTensor
            The global tensor view created.
        """
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
        src: GlobalTensor,
        /,
        *,
        offsets: Sequence[Expr | int],
        shape: Optional[Sequence[int]] = None,
        layout: Optional[RegisterLayout] = None,
        dims: Optional[Sequence[int]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Load a slice of global tensor into a register tensor.

        This instruction loads a slice of the global tensor `x` into a register tensor, given the `offsets` for each
        dimension of the global tensor and the `shape` of the slice to be loaded.

        When we only slice over a subset of the dimensions of the global tensor, we can specify the `dims` parameter to
        indicate which dimensions are being sliced.

        The optional `layout` parameter can be used to specify the layout of the register tensor.

        When `out` is provided, the loaded data will be stored in the `out` register tensor, otherwise a new register
        tensor will be allocated.

        Parameters
        ----------
        src: GlobalTensor
            The global tensor to load from.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor. The length of this sequence must match the number
            of dimensions of the global tensor.
        shape: Sequence[int], optional
            The shape of the slice to be loaded. If not provided, the shape of the global tensor will be used.
        layout: RegisterLayout, optional
            The layout of the register tensor. If not provided, the layout will be inferred based on the operations
            performed on it. When provided, its shape must match the `shape` parameter.
        dims: Sequence[int], optional
            The dimensions of the global tensor that are being sliced. If not provided, it is assumed that all
            dimensions are being sliced. The length of this sequence must match the number of dimensions of the
            register tensor being loaded into.
        out: RegisterTensor, optional
            The register tensor to store the loaded data into. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the loaded data from the global tensor.
        """
        if len(offsets) != len(src.shape):
            raise InstructionError(
                "The number of offsets must be equal to the number of dimensions of the global tensor"
            )
        return self._builder.load_global(x=src, offsets=offsets, dims=dims, shape=shape, layout=layout, output=out)

    def store_global(
        self,
        dst: GlobalTensor,
        src: RegisterTensor,
        *,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        """Store a register tensor into a slice of a global tensor.

        This instruction stores the contents of the register tensor `x` into a slice of the global tensor `dst`.

        The `offsets` parameter specifies the starting offsets for each dimension of the global tensor where the
        register tensor will be stored. The length of this sequence must match the number of dimensions of the global
        tensor.

        The `dims` parameter specifies which dimensions of the global tensor are being sliced. The dimension dim[0] of
        the global tensor corresponds to the first dimension of the register tensor, dim[1] to the second, and so on.
        If `dims` is not provided, it is assumed to be range(len(dst.shape)), meaning all dimensions of the global tensor
        are being sliced in the same order as the register tensor. When provided, the length of this sequence must
        match the number of dimensions of the register tensor being stored.

        Parameters
        ----------
        dst: GlobalTensor
            The global tensor to store into.
        src: RegisterTensor
            The register tensor to store into the global tensor.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor where the register tensor will be stored.
        dims: Sequence[int], optional
            The dimensions of the global tensor that are being sliced.
        """
        if dims is not None and len(dims) != len(src.shape):
            raise InstructionError(
                "The number of slice dimensions must be equal to the number of dimensions of the "
                f"register tensor: {len(dims)} vs {len(src.shape)}"
            )
        return self._builder.store_global(dst=dst, src=src, offsets=offsets, dims=dims)

    def load_shared(
        self,
        src: SharedTensor,
        *,
        layout: Optional[RegisterLayout] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Load a shared tensor into a register tensor.

        Parameters
        ----------
        src: SharedTensor
            The shared tensor to load from.
        layout: RegisterLayout, optional
            The layout of the register tensor. If not provided, the layout will be inferred based on the operations
            performed on it. When provided, its shape must match the shape of the shared tensor.
        out: RegisterTensor, optional
            The register tensor to store the loaded data into. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the loaded data from the shared tensor.
        """
        return self._builder.load_shared(src=src, layout=layout, output=out)

    def store_shared(
        self,
        dst: SharedTensor,
        src: RegisterTensor,
        *,
        offsets: Optional[Sequence[int]] = None,
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        """Store a register tensor into a shared tensor.

        This instruction stores the contents of the register tensor `src` into a slice of the shared tensor `dst`.

        Parameters
        ----------
        dst: SharedTensor
            The shared tensor to store into.
        src: RegisterTensor
            The register tensor to store into the shared tensor.
        offsets: Sequence[int], optional
            The offsets for each dimension of the shared tensor where the register tensor will be stored.
        dims: Sequence[int], optional
            The dimensions of the shared tensor that are being sliced. If not provided, it is assumed that all
            dimensions are being sliced in the same order as the register tensor. The length of this sequence must
            match the number of dimensions of the register tensor being stored.
        """
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
        """Free a shared tensor.

        Parameters
        ----------
        tensor: SharedTensor
            The shared tensor to free.
        """
        self._builder.free_shared(tensor)

    def copy_async(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> None:
        if dims is None:
            if len(dst.shape) != len(src.shape):
                raise InstructionError(
                    "The number of dimensions of the source global tensor must match the destination shared tensor if dims is not specified"
                )
        if len(offsets) != len(src.shape):
            raise InstructionError(
                "The number of offsets must be equal to the number of dimensions of the source global tensor"
            )
        self._builder.copy_async(dst=dst, src=src, offsets=offsets, dims=dims, evict=evict, check_bounds=check_bounds)

    def copy_async_wait_all(self):
        self._builder.copy_async_wait_all()

    def copy_async_commit_group(self):
        self._builder.copy_async_commit_group()

    def copy_async_wait_group(self, n: Union[Expr, int]) -> None:
        self._builder.copy_async_wait_group(n)

    def dot(
        self,
        a: RegisterTensor,
        b: RegisterTensor,
        c: Optional[RegisterTensor] = None,
        /,
        *,
        acc_dtype: Optional[DataType] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Dot product.

        This instruction computes the dot product: `out = a @ b + c`.

        The `a`, `b` and (optional) `c` tensors must be 2D register tensors, where

        - `a` has shape [m, k]
        - `b` has shape [k, n]
        - `c` has shape [m, n]

        If `c` is not provided, it's assumed to be a zero-initialized accumulator tensor with `acc_dtype` as its data
        type. If both `c` and `acc_dtype` are not provided, an error is raised.

        The `out` tensor is optional. If provided, it will be used to store the result of the dot product. If not
        provided, a new register tensor will be allocated to hold the result.

        The data type of the `c` and `out` must be the same and match the `acc_dtype` if they are provided.

        Parameters
        ----------
        a: RegisterTensor
            The first input tensor with shape [m, k].
        b: RegisterTensor
            The second input tensor with shape [k, n].
        c: RegisterTensor, optional
            The accumulator tensor with shape [m, n]. If not provided, a zero-initialized tensor will be used.
        acc_dtype: DataType, optional
            The data type of the accumulation computation. If `c` is not provided, this is used to determine the
            data type of the `c` tensor. If `c` is provided, it must match the data type of `c`.
        out: RegisterTensor, optional
            The output tensor to store the result of the dot product. If not provided, a new register tensor will be
            allocated to hold the result.

        Returns
        -------
        ret: RegisterTensor
            The result of the dot product, which is a register tensor with shape [m, n]. It will be `out` if provided,
            or a new register tensor if not.
        """
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
        if not (len(a.shape) == len(b.shape) == len(c.shape) == 2):
            raise InstructionError("mma_dot requires 2D tensors for a, b, and c")
        if a.shape[1] != b.shape[0] or a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
            raise InstructionError(
                "The shapes of a, b, and c must match for dot: "
                f"a: {a.shape}, b: {b.shape}, c: {c.shape} (expected a.shape[1] == b.shape[0] and a.shape[0] == c.shape[0] and b.shape[1] == c.shape[1])"
            )
        return self._builder.dot(
            a,
            b,
            c,
            output=out,
        )

    def cast(self, x: RegisterTensor, dtype: DataType) -> RegisterTensor:
        """Cast a register tensor to a different data type.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to be cast.
        dtype: DataType
            The target data type to cast the register tensor to.

        Returns
        -------
        ret: RegisterTensor
            The register tensor with the specified data type.
        """
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

    def exp2(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.exp2(x, out=out)

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
        shape: Sequence[int],
        ptr: Var,
        f_offset: Callable[..., Expr | int],
        f_mask: Optional[Callable[..., Expr | int | bool]] = None,
        layout: Optional[RegisterLayout] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.load_global_generic(
            dtype=dtype,
            shape=shape,
            ptr=ptr,
            f_offset=lambda args: f_offset(*args),
            f_mask=lambda args: f_mask(*args) if f_mask is not None else None,
            layout=layout,
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
        """Perform a synchronization.

        The thread block will continue execution only after all previous instructions finished executing.
        """
        self._builder.syncthreads()

    def annotate_layout(self, tensor: RegisterTensor, layout: RegisterLayout) -> None:
        """
        Annotate the layout of a register tensor.

        Parameters
        ----------
        tensor: RegisterTensor
            The tensor to annotate.
        layout: RegisterLayout
            The layout to annotate the tensor with.
        """
        self._builder.annotate_layout(tensor, layout)

    def print_tensor(self, msg: str, tensor: Tensor, fmt: Optional[str] = None) -> None:
        self._builder.print_tensor(msg=msg, tensor=tensor, fmt=fmt)

    def printf(self, fstring: str, *args: Expr | int | float) -> None:
        self._builder.printf(fstring, *args)

    def assign(self, dst: RegisterTensor, src: RegisterTensor) -> None:
        """
        Assign the value of src to dst.

        Parameters
        ----------
        dst: RegisterTensor
            The destination tensor.
        src: RegisterTensor
            The source tensor.
        """
        if dst.dtype != src.dtype:
            raise InstructionError("The dtypes of dst and src must match, got {} and {}".format(dst.dtype, src.dtype))
        self._builder.assign_register(dst, src)

    @staticmethod
    def static_assert(cond: bool | Expr, msg: str) -> None:
        if not isinstance(cond, Constant) and not isinstance(cond, bool):
            raise ValueError("Static assert condition must be a constant")
        if not cond:
            raise AssertionError(msg)


def autotune(arg_names: str, arg_values: Sequence[Any]) -> Callable[[Type[Script]], Any]:
    """Annotate an autotune subspace for a tilus script.

    Parameters
    ----------
    arg_names: str
        The names of the arguments for autotuning, separated by commas.
    arg_values: Sequence[Any]
        The sequence of the choices for the autotune parameters. Each choice can be a single value or a sequence of
        values that match the names in `arg_names`.

    Returns
    -------
    ret: Callable[[Type[Script]], Type[Script]]
        The decorator that can be applied to a tilus script class for the marking of autotune parameters.
    """

    def decorator(script_cls):
        if not hasattr(script_cls, "_autotune_space"):
            script_cls._autotune_space = {}
        space = getattr(script_cls, "_autotune_space")
        names = [name.strip() for name in arg_names.split(",")]

        # check names and arg_values
        # 1. can not define the same name more than once
        if any(name in space for name in names):
            common_names = set(names) & set(space.keys())
            raise RuntimeError("Duplicated specification for parameters: {}".format(common_names))
        # 2. the arg_values should match the names during unpacking
        if not isinstance(arg_values, Sequence):
            raise TypeError("The arg_values values must be a sequence")
        for arg_value in arg_values:
            if len(names) > 1:
                if not isinstance(arg_value, Sequence) or len(arg_value) != len(names):
                    raise TypeError(
                        "Can not unpack the arg_values for arg_names\n"
                        f"  arg_names: {arg_names}\n"
                        f"  arg_value: {arg_value}"
                    )

        space[arg_names] = arg_values
        setattr(script_cls, "_autotune_space", space)

        # return functools.wraps(wrapped=script_cls, assigned=arg_names)(script_cls)
        return script_cls

    return decorator
