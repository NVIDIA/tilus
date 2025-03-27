from __future__ import annotations

import typing
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Type, Union

from hidet.ir.expr import Expr, Var
from hidet.ir.primitives.cuda.vars import blockIdx, dim3, gridDim
from hidet.ir.type import DataType
from tilus.ir.builders import StmtBuilder
from tilus.ir.instructions import MmaDotConfig
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
        from tilus.lang.instantiated_script import InstantiatedScript

        script_cls: Type[Script] = cls
        return InstantiatedScript(
            script_cls=script_cls,
            script_args=args,
            script_kwargs=kwargs,
        )

    def __init__(self) -> None:
        # builder used to append instructions
        self._builder: Optional[StmtBuilder] = None

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

        if shape is None and layout is None:
            raise ValueError("Must specify either shape or layout")
        elif shape is not None and layout is not None:
            raise ValueError("Cannot specify both shape and layout")
        elif layout is None:
            layout = self.cuda.default_register_layout(num_warps=self.attrs.warps, dtype=dtype, shape=shape)

        return self._builder.allocate_register(dtype=dtype, layout=layout, f_init=f_init)

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
        from tilus.ir.layout.shared_layout import shared_repeat

        match (shape, layout):
            case (None, None):
                raise ValueError("Must specify either shape or layout")
            case (_, None):
                assert isinstance(shape, Sequence)
                layout = shared_repeat(*shape)
            case (None, _):
                pass
            case _:
                raise ValueError("Cannot specify both shape and layout")

        assert layout is not None
        return self._builder.allocate_shared(dtype=dtype, shared_layout=layout)

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
        offsets: Sequence[Expr],
        shape: Optional[Sequence[int]] = None,
        layout: Optional[RegisterLayout] = None,
        dims: Optional[Sequence[int]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        match (shape, layout, out):
            case (None, None, None):
                raise ValueError("Must specify one of shape, layout and out")
            case (_, None, None):
                from tilus.ir.layout import auto_repeat_spatial

                if self.attrs.warps is None:
                    raise ValueError(
                        "Must specify the number of warps in th script so that load_global could use it "
                        "to infer the register tensor layout"
                    )
                layout = auto_repeat_spatial(num_threads=self.attrs.warps * 32, shape=shape)  # type: ignore[arg-type]
                out = RegisterTensor.create(dtype=x.dtype, layout=layout)
            case (None, _, None):
                out = RegisterTensor.create(dtype=x.dtype, layout=layout)  # type: ignore[arg-type]
            case (None, None, _):
                pass
            case _:
                raise ValueError("Cannot specify any two of shape, layout, and out")

        return self._builder.load_global(x=x, offsets=offsets, dims=dims, output=out)

    def store_global(
        self,
        dst: GlobalTensor,
        x: RegisterTensor,
        *,
        offsets: Sequence[Expr],
        slice_dims: Optional[Sequence[int]] = None,
    ) -> None:
        self._builder.store_global(dst=dst, src=x, offsets=offsets, dims=slice_dims)

    def load_shared(
        self,
        src: SharedTensor,
        *,
        offsets: Optional[Sequence[Expr | int]] = None,
        dims: Optional[Sequence[int]] = None,
        out_layout: Optional[RegisterLayout] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if offsets is not None:
            if dims is None:
                dims = list(range(len(src.shape)))
            src = self._builder.shared_slice(src, offsets=offsets, slice_dims=dims, slice_shape=out_layout.shape)
        if out_layout is None and out is None:
            out_layout = self.cuda.default_register_layout(num_warps=self.attrs.warps, dtype=src.dtype, shape=src.shape)
        return self._builder.load_shared(src=src, output_layout=out_layout, output=out)

    def store_shared(
        self,
        dst: SharedTensor,
        src: RegisterTensor,
        *,
        offsets: Optional[Sequence[int]] = None,
        dims: Optional[Sequence[int]] = None,
    ) -> None:
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
    ) -> None:
        self._builder.copy_async(dst=dst, src=src, offsets=offsets, dims=dims, evict=evict)

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
        c: RegisterTensor,
        /,
        *,
        config: MmaDotConfig,
        warp_spatial: Sequence[int],
        warp_repeat: Sequence[int],
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        return self._builder.mma_dot(
            a, b, c, config=config, warp_spatial=warp_spatial, warp_repeat=warp_repeat, output=output
        )

    def cast(self, x: RegisterTensor, dtype: DataType) -> RegisterTensor:
        return self._builder.cast(x=x, dtype=dtype)

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
