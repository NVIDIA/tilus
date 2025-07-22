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
from typing import List

from tilus.ir.func import Function
from tilus.ir.functors import IRVisitor
from tilus.ir.inst import Instruction


class InstructionCollector(IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.instructions: List[Instruction] = []

    def visit_Instruction(self, inst: Instruction) -> None:
        self.instructions.append(inst)


def collect_instructions(prog: Function) -> List[Instruction]:
    collector = InstructionCollector()
    collector(prog)
    return collector.instructions
