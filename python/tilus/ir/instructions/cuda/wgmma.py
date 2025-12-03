from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from hidet.ir.expr import Constant, Expr
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor 

@dataclass(frozen=True, eq=False)
class WgmmaFenceInst(Instruction):
    @staticmethod
    def create() -> WgmmaFenceInst:
        return WgmmaFenceInst(output=None, inputs=())

@dataclass(frozen=True, eq=False)
class WgmmaCommitGroupInst(Instruction):
    @staticmethod
    def create() -> WgmmaCommitGroupInst:
        return WgmmaCommitGroupInst(output=None, inputs=())

@dataclass(frozen=True, eq=False)
class WgmmaWaitGroupInst(Instruction):
    n: Expr
    @staticmethod
    def create(n: Expr) -> WgmmaWaitGroupInst:
        return WgmmaWaitGroupInst(output=None, inputs=(), n=n)

@dataclass(frozen=True, eq=False)
class WgmmaMmaSSInst(Instruction):
    @staticmethod
    def create(a: SharedTensor, b: SharedTensor, d: RegisterTensor) -> WgmmaMmaSSInst:
        return WgmmaMmaSSInst(output=None, inputs=(a, b, d))

@dataclass(frozen=True, eq=False)
class WgmmaMmaRSInst(Instruction):
    @staticmethod
    def create(a: RegisterTensor, b: SharedTensor, d: RegisterTensor) -> WgmmaMmaRSInst:
        return WgmmaMmaRSInst(output=None, inputs=(a, b, d))