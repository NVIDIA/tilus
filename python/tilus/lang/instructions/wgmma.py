import contextlib
from typing import Optional, Sequence, Union

from hidet.ir.expr import Expr
from hidet.ir.type import DataType

from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor

from .root import InstructionGroup


class WgmmaInstructionGroup(InstructionGroup):
    def fence(self) -> None:
        self._builder.wgmma_fence()

    def commit_group(self) -> None:
        self._builder.wgmma_commit_group()

    def wait_group(self, n: Union[Expr, int]) -> None:
        self._builder.wgmma_wait_group(n)

    def mma(self, a: SharedTensor | RegisterTensor, b: SharedTensor, d: RegisterTensor) -> None:
        if isinstance(a, SharedTensor):
            if len(a.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(a.shape))
            if len(b.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(b.shape))
            if len(d.shape) != 2:
                raise InstructionError("mma requires 2D register tensors, got shape {}".format(d.shape))
            self._builder.wgmma_mma_ss(a, b, d)
        elif isinstance(a, RegisterTensor):
            if len(a.shape) != 2:
                raise InstructionError("mma requires 2D register tensors, got shape {}".format(a.shape))
            if len(b.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(b.shape))
            if len(d.shape) != 2:
                raise InstructionError("mma requires 2D register tensors, got shape {}".format(d.shape))
            self._builder.wgmma_mma_rs(a, b, d)
        else:
            raise InstructionError("Invalid type of a: {}, expected SharedTensor or RegisterTensor".format(type(a)))