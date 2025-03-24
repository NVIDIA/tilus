from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import GlobalViewInst


@register_emitter(GlobalViewInst)
class GlobalViewInstEmitter(BaseInstEmitter):
    def emit(self, inst: GlobalViewInst) -> None:
        self.assign(self.get_or_allocate_var(inst.global_output), inst.ptr)
