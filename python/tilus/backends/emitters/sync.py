from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import SyncReduceThreadsInst, SyncThreadsInst


@register_emitter(SyncThreadsInst)
class SyncThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncThreadsInst) -> None:
        self.sync()


@register_emitter(SyncReduceThreadsInst)
class SyncReduceThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncReduceThreadsInst) -> None:
        self.declare(inst.var, init=self.sync_reduce(inst.reduce_value, op=inst.reduce_op))
