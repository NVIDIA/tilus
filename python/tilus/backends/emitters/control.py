from tilus.extensions.hidet.ir.primitives.cuda.control import exit
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.instructions import ExitInst
from tilus.target import nvgpu_any


@register_inst_emitter(ExitInst, target=nvgpu_any)
class ExitInstEmitter(BaseInstEmitter):
    def emit(self, inst: ExitInst):  # type: ignore[override]
        self.append(exit())
