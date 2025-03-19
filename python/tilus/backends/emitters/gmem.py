from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import GlobalViewInst
from tilus.target import gpgpu_any


@register_inst_emitter(GlobalViewInst, target=gpgpu_any)
class GlobalViewInstEmitter(BaseInstEmitter):
    def emit(self, inst: GlobalViewInst) -> None:
        self.assign(self.get_or_allocate_var(inst.global_output), inst.ptr)
