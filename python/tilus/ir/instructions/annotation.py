from __future__ import annotations

from dataclasses import dataclass

from tilus.ir.inst import Instruction
from tilus.ir.layout import RegisterLayout
from tilus.ir.tensor import RegisterTensor, Tensor


@dataclass(frozen=True, eq=False)
class AnnotateLayoutInst(Instruction):
    layout: RegisterLayout

    @staticmethod
    def create(tensor: Tensor, layout: RegisterLayout) -> AnnotateLayoutInst:
        assert isinstance(tensor, RegisterTensor), tensor
        return AnnotateLayoutInst(output=None, inputs=(tensor,), layout=layout)
