from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import equal
from hidet.ir.primitives.cuda.ldst import load, store

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import LockSemaphoreInst, ReleaseSemaphoreInst


@register_emitter(LockSemaphoreInst)
class LockSemaphoreEmitter(BaseInstEmitter):
    def emit(self, inst: LockSemaphoreInst) -> None:
        semaphore = self.declare_var("semaphore", tp=~int32, init=inst.semaphore)
        semaphore_expect = self.declare_var("semaphore_expect", tp=int32, init=inst.value)

        with self.while_loop(boolean.true):
            semaphore_value = self.declare_var("semaphore_value", tp=int32, init=-int32.one)
            with self.if_then(self.current_worker == 0):
                self.assign(semaphore_value, load(addr=semaphore, space="generic", sync="acquire", scope="gpu"))
            cond = self.sync_reduce(equal(semaphore_value, semaphore_expect), op="or")
            with self.if_then(cond):
                self.brk()


@register_emitter(ReleaseSemaphoreInst)
class ReleaseSemaphoreEmitter(BaseInstEmitter):
    def emit(self, inst: ReleaseSemaphoreInst) -> None:
        semaphore = self.declare_var("semaphore", tp=~int32, init=inst.semaphore)

        with self.if_then(self.current_worker == 0):
            self.append(store(addr=semaphore, space="generic", value=inst.value, sync="release", scope="gpu"))
