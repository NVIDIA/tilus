from __future__ import annotations
from typing import List, Sequence
from pathlib import Path
from tilus.ir.prog import Program
from tilus.ir.func import Function
from tilus.transforms.instruments import PassInstrument, DumpIRInstrument


class PassContext:
    _ctx = None

    def __init__(self) -> None:
        self.instruments: List[PassInstrument] = []

    def __enter__(self):
        assert PassContext._ctx is None
        PassContext._ctx = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return None
        assert PassContext._ctx is self
        PassContext._ctx = None

    def before_all_passes(self, program: Program) -> None:
        for instrument in self.instruments:
            instrument.before_all_passes(program)

    def before_pass(self, pass_name: str, program: Program) -> None:
        for instrument in self.instruments:
            instrument.before_pass(pass_name, program)

    def after_pass(self, pass_name: str, program: Program) -> None:
        for instrument in self.instruments:
            instrument.after_pass(pass_name, program)

    def after_all_passes(self, program: Program) -> None:
        for instrument in self.instruments:
            instrument.after_all_passes(program)

    def dump_ir(self, dump_dir: Path) -> None:
        self.instruments.append(DumpIRInstrument(dump_dir))

    @staticmethod
    def current() -> PassContext:
        if PassContext._ctx is None:
            return PassContext()  # use a default context
        else:
            return PassContext._ctx


class Pass:
    def __init__(self) -> None:
        self.name: str = self.__class__.__name__.removesuffix("Pass")

    def __call__(self, prog: Program) -> Program:
        return self.process_program(prog)

    def process_program(self, program: Program) -> Program:
        functions: dict[str, Function] = {name: self.process_function(func) for name, func in program.functions.items()}
        if all(a is b for a, b in zip(functions.values(), program.functions.values())):
            return program
        else:
            return Program.create(functions)

    def process_function(self, function: Function) -> Function:
        raise NotImplementedError()


def apply_transforms(prog: Program, transforms: Sequence[Pass]) -> Program:
    """
    Apply the given sequence of transforms to the program.

    The instruments in the current PassContext are performed.

    Parameters
    ----------
    prog: Program
        The program to transform.

    transforms: Sequence[Pass]
        The sequence of transforms to apply.

    Returns
    -------
    prog: Program
        The transformed program.
    """
    ctx = PassContext.current()
    ctx.before_all_passes(prog)
    for transform in transforms:
        ctx.before_pass(transform.name, prog)
        prog = transform(prog)
        ctx.after_pass(transform.name, prog)
    ctx.after_all_passes(prog)
    return prog
