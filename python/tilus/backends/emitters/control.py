from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.control import exit
from tilus.ir.instructions import ExitInst


@register_emitter(ExitInst)
class ExitInstEmitter(BaseInstEmitter):
    def emit(self, inst: ExitInst) -> None:
        self.append(exit())
