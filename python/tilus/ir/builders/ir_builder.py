from __future__ import annotations

from tilus.ir.builders.func_builder import FunctionBuilder
from tilus.ir.func import Function
from tilus.ir.prog import Program


class IRBuilder(FunctionBuilder):
    class _ProgramContext:
        def __init__(self, ib: IRBuilder) -> None:
            self.ib: IRBuilder = ib

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

    def __init__(self) -> None:
        super().__init__()
        self._built_program: Program = Program.create(functions={})

    def _on_finish(self, built_function: Function) -> None:
        super()._on_finish(built_function)
        if built_function.name in self._built_program.functions:
            raise ValueError(f"Function {built_function.name} already exists in the program")
        self._built_program = self._built_program.with_function(built_function)

    def program(self) -> _ProgramContext:
        return self._ProgramContext(self)

    def flush_program(self) -> Program:
        return self._built_program
