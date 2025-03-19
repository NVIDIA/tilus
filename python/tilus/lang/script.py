import typing
from types import FunctionType
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Union

from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType
from hidet.runtime.compiled_module import CompiledModule
from tilus.drivers import BuildOptions, build_program
from tilus.ir.inst import (
    AllocateRegisterInst,
    CastInst,
    FormatPrintInst,
    GlobalViewInst,
    Instruction,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    MmaDotInst,
    PrintValueInst,
    StoreGlobalGenericInst,
    StoreGlobalInst,
)
from tilus.ir.layout import GlobalLayout, RegisterLayout, global_repeat, global_strides
from tilus.ir.prog import Program
from tilus.ir.stmt import InstStmt, Stmt
from tilus.ir.tensor import GlobalTensor, RegisterTensor, Tensor
from tilus.lang.modules.cuda import cuda
from tilus.lang.modules.utils import utils
from tilus.lang.utils import group_function_argument


class CompiledScript:
    def __init__(self) -> None:
        self._program: Optional[Program] = None
        self._compiled_module: Optional[CompiledModule] = None
        self._kernels: set[str] = set()

        self._detect_kernels()

    def __getattribute__(self, item):
        if item in super().__getattribute__("_kernels"):
            return self.compiled()[item]
        else:
            return super().__getattribute__(item)

    def _detect_kernels(self) -> None:
        # get the method list unique to the current class
        current_class = self.__class__
        script_class = Script
        unique_attrs: set[str] = set(dir(current_class)) - set(dir(script_class))
        unique_methods: list[str] = [name for name in unique_attrs if callable(getattr(current_class, name))]

        # if the method is a function, add it to the kernel list with a None value
        kernels = set()
        for unique_method in unique_methods:
            if isinstance(getattr(current_class, unique_method), FunctionType):
                kernels.add(unique_method)
        self._kernels = kernels

    def compiled(self) -> CompiledModule:
        raise NotImplementedError()


class ScriptTracer:
    def __init__(self) -> None:
        from tilus.lang.transpiler import Transpiler

        self._transpiler: Optional[Transpiler] = None

    def _append(self, inst_or_stmt: Union[Instruction, Stmt]) -> None:
        assert self._transpiler is not None

        stmt = inst_or_stmt if isinstance(inst_or_stmt, Stmt) else InstStmt(inst=inst_or_stmt)
        self._transpiler.current_scope.stmts.append(stmt)


class Attributes:
    """
    Attributes of the script program.
    """

    """
    The number of blocks. 
    """
    blocks: list[Any] | Any

    """
    The number of warps.
    """
    warps: int


class Dim3:
    """
    The three dimensions of the grid of blocks.
    """

    x: Var
    y: Var
    z: Var


class Script(CompiledScript, ScriptTracer):
    def __init__(self, debug_block: Optional[tuple[int, ...] | int] = None) -> None:
        super().__init__()
        self._debug_block = debug_block
        # the following attributes should be set by the user in the kernel function
        self.attrs: Attributes = Attributes()
        self.blockIdx: Dim3 = Dim3()

        # the following primitives could be used in the __init__ function to prepare the layouts
        self.cuda = cuda
        self.utils = utils

    def __call__(self, *args):
        return self.compiled()(*args)

    def program(self):
        """
        Get the tilus.ir.Program object from the script.

        Returns
        -------
        prog: Program
            The corresponding program of this script.
        """
        from tilus.lang.transpiler import Transpiler

        if self._program is None:
            functions = {}
            for kernel in self._kernels:
                transpiler = Transpiler()
                functions[kernel] = transpiler.transpile(script=self, method=getattr(self.__class__, kernel))
            self._program = Program.create(functions=functions)

        return self._program

    def compiled(self) -> CompiledModule:
        """
        Get the compiled module of this script.

        Returns
        -------
        module: CompiledModule
            The compiled module.
        """
        if self._compiled_module is None:
            self._compiled_module = build_program(self.program(), options=BuildOptions.create(self._debug_block))
        return self._compiled_module

    # the following functions should be called in the kernel function

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

    def load_global_flex(
        self,
        *,
        dtype: DataType,
        layout: RegisterLayout,
        ptr: Var,
        f_offset: Callable[..., Expr | int],
        f_mask: Optional[Callable[..., Expr | int | bool]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = LoadGlobalGenericInst.create(
            dtype=dtype,
            layout=layout,
            ptr=ptr,
            f_offset=group_function_argument(f_offset),
            f_mask=group_function_argument(f_mask) if f_mask else None,
            out=out,
        )
        self._append(inst)
        return inst.register_output

    def store_global_flex(
        self,
        x: RegisterTensor,
        /,
        *,
        ptr: Var,
        f_offset: Callable[..., Expr | int],
        f_mask: Optional[Callable[..., Expr | int | bool]] = None,
    ) -> None:
        inst = StoreGlobalGenericInst.create(
            x=x,
            ptr=ptr,
            f_offset=group_function_argument(f_offset),
            f_mask=group_function_argument(f_mask) if f_mask else None,
        )
        self._append(inst)

    def global_view(
        self,
        ptr: Expr,
        *,
        dtype: DataType,
        shape: Optional[Sequence[Expr | int]] = None,
        strides: Optional[Sequence[Expr | int]] = None,
        layout: Optional[GlobalLayout] = None,
    ) -> GlobalTensor:
        global_layout: GlobalLayout
        if layout is not None:
            assert shape is None and strides is None, "Cannot specify both layout and shape/strides"
            global_layout = layout
        else:
            assert shape is not None, "Must specify shape when layout is not provided"
            if strides is None:
                # assume compact row-major layout
                global_layout = global_repeat(*shape)
            else:
                assert len(shape) == len(strides), "Shape and strides must have the same length"
                global_layout = global_strides(shape, strides)

        inst = GlobalViewInst.create(dtype=dtype, layout=global_layout, ptr=ptr)
        self._append(inst)
        return inst.global_output

    def load_global(
        self,
        x: GlobalTensor,
        /,
        *,
        offsets: Sequence[Expr],
        layout: Optional[RegisterLayout] = None,
        dims: Optional[Iterable[int]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = LoadGlobalInst.create(x, offsets=offsets, dims=dims, layout=layout, out=out)
        self._append(inst)
        return inst.register_output

    def store_global(
        self,
        dst: GlobalTensor,
        x: RegisterTensor,
        *,
        offsets: Sequence[Expr],
        slice_dims: Optional[Sequence[int]] = None,
    ) -> None:
        inst = StoreGlobalInst.create(dst, x, offsets=offsets, dims=slice_dims)
        self._append(inst)

    def register_tensor(
        self, dtype: DataType, layout: RegisterLayout, f_init: Optional[Callable[[Sequence[Var]], Expr]] = None
    ) -> RegisterTensor:
        inst = AllocateRegisterInst.create(dtype=dtype, layout=layout, f_init=f_init)
        self._append(inst)
        return inst.register_output

    def mma_dot(
        self,
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        /,
        *,
        mma_inst: str,
        warp_spatial: tuple[int, int, int],
        warp_repeat: tuple[int, int, int],
        output: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        inst = MmaDotInst.create(
            a=a, b=b, c=c, mma_inst=mma_inst, warp_spatial=warp_spatial, warp_repeat=warp_repeat, output=output
        )
        self._append(inst)
        return inst.register_output

    def cast(
        self,
        x: RegisterTensor,
        /,
        dtype: DataType,
        *,
        interleave_width: Optional[int] = None,
        interleave_stride: Optional[int] = None,
        ignore_int4b_xor: bool = False,
    ) -> RegisterTensor:
        inst = CastInst.create(
            dtype=dtype,
            x=x,
            interleave_width=interleave_width,
            interleave_stride=interleave_stride,
            ignore_int4b_xor=ignore_int4b_xor,
        )
        self._append(inst)
        return inst.register_output

    def print_tensor(self, msg: str, x: Tensor, fmt: Optional[str] = None) -> None:
        inst = PrintValueInst.create(x, cond=boolean.true, msg=msg, fmt=fmt)
        self._append(inst)

    def printf(self, fstring: str, *args: Expr | int | float) -> None:
        inst = FormatPrintInst.create(cond=boolean.true, fstring=fstring, expressions_=args)
        self._append(inst)
