from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import SyncReduceThreadsInst, SyncThreadsInst
from tilus.target import gpgpu_any, nvgpu_any


@register_inst_emitter(SyncThreadsInst, target=gpgpu_any)
class SyncThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncThreadsInst) -> None:
        self.sync()


@register_inst_emitter(SyncReduceThreadsInst, target=nvgpu_any)
class SyncReduceThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncReduceThreadsInst) -> None:
        self.declare(inst.var, init=self.sync_reduce(inst.reduce_value, op=inst.reduce_op))
