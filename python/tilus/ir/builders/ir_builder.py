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

from tilus.ir.builders.func_builder import FunctionBuilder
from tilus.ir.func import Function
from tilus.ir.prog import Program


class IRBuilder(FunctionBuilder):
    class _ProgramContext:
        def __init__(self, ib: IRBuilder) -> None:
            self.ib: IRBuilder = ib

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

    def __init__(self) -> None:
        super().__init__()
        self._built_program: Program = Program.create(functions={})

    def _on_finish(self, built_function: Function) -> None:
        super()._on_finish(built_function)
        if built_function.name in self._built_program.functions:
            raise ValueError(f"Function {built_function.name} already exists in the program")
        self._built_program = self._built_program.with_function(built_function)

    def program(self) -> _ProgramContext:
        return self._ProgramContext(self)

    def flush_program(self) -> Program:
        return self._built_program
