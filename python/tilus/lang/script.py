from typing import Any, Sequence, Union, Optional, Callable, List
from types import FunctionType
from hidet.ir.type import DataType
from hidet.ir.expr import Var, Expr
from hidet.runtime.compiled_module import CompiledModule
from tilus.ir.layout import Layout
from tilus.ir.stmt import Stmt, InstructionStmt
from tilus.ir.value import RegisterValue
from tilus.ir.inst import Instruction, LoadGlobalInst, StoreGlobalInst
from tilus.drivers import build_program
from tilus.ir.prog import Program


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

        stmt = inst_or_stmt if isinstance(inst_or_stmt, Stmt) else InstructionStmt(inst=inst_or_stmt)
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
    def __init__(self) -> None:
        super().__init__()
        self.attrs: Attributes = Attributes()
        self.blockIdx: Dim3 = Dim3()

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
            self._compiled_module = build_program(self.program())
        return self._compiled_module

    def load_global(
        self,
        *,
        dtype: DataType,
        layout: Layout,
        ptr: Var,
        f_offset: Callable[[Sequence[Var]], Expr | int],
        f_mask: Optional[Callable[[Sequence[Var]], Expr | int | bool]] = None,
        out: Optional[RegisterValue] = None,
    ) -> RegisterValue:
        inst = LoadGlobalInst.create(dtype=dtype, layout=layout, ptr=ptr, f_offset=f_offset, f_mask=f_mask, out=out)
        self._append(inst)
        return inst.register_output

    def store_global(
        self,
        x: RegisterValue,
        *,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr | int],
        f_mask: Optional[Callable[[List[Var]], Expr | int | bool]] = None,
    ) -> None:
        inst = StoreGlobalInst.create(x=x, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        self._append(inst)
