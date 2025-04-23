from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from tilus.ir.node import IRNode
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor


@dataclass(frozen=True, eq=False)
class Instruction(IRNode):
    output: Optional[Tensor]
    inputs: tuple[Tensor, ...]

    @property
    def shared_output(self) -> SharedTensor:
        assert isinstance(self.output, SharedTensor), self.output
        return self.output

    @property
    def register_output(self) -> RegisterTensor:
        assert isinstance(self.output, RegisterTensor), self.output
        return self.output

    @property
    def register_or_shared_output(self) -> SharedTensor | RegisterTensor:
        assert isinstance(self.output, SharedTensor) or isinstance(self.output, RegisterTensor), self.output
        return self.output

    @property
    def global_output(self) -> GlobalTensor:
        assert isinstance(self.output, GlobalTensor), self.output
        return self.output

    @property
    def register_input(self) -> RegisterTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, RegisterTensor)
        return x

    @property
    def shared_input(self) -> SharedTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, SharedTensor)
        return x

    @property
    def attributes(self) -> Mapping[str, Any]:
        attrs = {}
        for k, v in self.__dict__.items():
            if k in ["output", "inputs"]:
                continue
            attrs[k] = v
        return attrs


@dataclass(frozen=True, eq=False)
class InstructionConfig(IRNode):
    pass


class InstructionError(Exception):
    """
    Exception raised when the parameters of an instruction are invalid.
    """
