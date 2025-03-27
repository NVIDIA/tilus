from hidet.ir.expr import Expr
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import AllocateGlobalInst, GlobalViewInst
from tilus.utils import cdiv


@register_emitter(GlobalViewInst)
class GlobalViewInstEmitter(BaseInstEmitter):
    def emit(self, inst: GlobalViewInst) -> None:
        self.assign(self.get_or_allocate_var(inst.global_output), inst.ptr)


@register_emitter(AllocateGlobalInst)
class AllocateGlobalInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateGlobalInst) -> None:
        tensor = inst.global_output
        ptr: Expr = self.codegen.allocate_global_memory(
            nbytes=cdiv(tensor.layout.size * tensor.dtype.nbits * 8, 8), clean=inst.require_clean
        )
        var = self.get_or_allocate_var(tensor)
        self.assign(var, ptr)
