from __future__ import annotations

from dataclasses import dataclass

from hidet.ir.expr import Expr

from tilus.ir.inst import Instruction


@dataclass(frozen=True, eq=False)
class LockSemaphoreInst(Instruction):
    semaphore: Expr
    value: Expr

    @staticmethod
    def create(
        semaphore: Expr,
        value: Expr,
    ) -> LockSemaphoreInst:
        return LockSemaphoreInst(inputs=(), output=None, semaphore=semaphore, value=value)


@dataclass(frozen=True, eq=False)
class ReleaseSemaphoreInst(Instruction):
    semaphore: Expr
    value: Expr

    @staticmethod
    def create(
        semaphore: Expr,
        value: Expr,
    ) -> ReleaseSemaphoreInst:
        return ReleaseSemaphoreInst(inputs=(), output=None, semaphore=semaphore, value=value)
