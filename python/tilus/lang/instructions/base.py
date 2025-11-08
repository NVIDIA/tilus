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
from typing import Optional

from tilus.ir.builders import StmtBuilder

_current_builder: Optional[StmtBuilder] = None


class InstructionBuilderContext:
    def __init__(self, builder: StmtBuilder) -> None:
        self.builder: StmtBuilder = builder

    def __enter__(self) -> None:
        global _current_builder
        _current_builder = self.builder

    def __exit__(self, exc_type, exc_value, traceback):
        global _current_builder
        _current_builder = None


class InstructionGroup:
    @property
    def _builder(self) -> StmtBuilder:
        global _current_builder
        assert _current_builder is not None
        return _current_builder


def builder_context(builder: StmtBuilder) -> InstructionBuilderContext:
    return InstructionBuilderContext(builder)
