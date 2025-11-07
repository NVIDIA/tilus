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


class InstructionGroup:
    def __init__(self):
        self._optinal_builder: Optional[StmtBuilder] = None

    def _set_builder(self, builder: Optional[StmtBuilder]) -> None:
        self._optional_builder = builder

    @property
    def _builder(self) -> StmtBuilder:
        if self._optional_builder is None:
            raise RuntimeError("Did you forget to call `super().__init__()` for the Tilus Script?")

        return self._optional_builder
