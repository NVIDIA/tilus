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


class InstructionGroup:
    @staticmethod
    def _set_builder(builder: Optional[StmtBuilder]) -> None:
        global _current_builder
        _current_builder = builder

    @property
    def _builder(self) -> StmtBuilder:
        global _current_builder
        assert _current_builder is not None
        return _current_builder
