# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tilus.ir.node import IRNode
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor, TMemoryTensor


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
    def tmemory_output(self) -> TMemoryTensor:
        assert isinstance(self.output, TMemoryTensor), self.output
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
    def global_input(self) -> GlobalTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, GlobalTensor)
        return x

    @property
    def register_or_shared_input(self) -> RegisterTensor | SharedTensor:
        assert len(self.inputs) == 1
        x = self.inputs[0]
        assert isinstance(x, RegisterTensor) or isinstance(x, SharedTensor)
        return x

    @property
    def attributes(self) -> dict[str, Any]:
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
