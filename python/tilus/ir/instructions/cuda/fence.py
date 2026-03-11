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

from tilus.ir.inst import Instruction


@dataclass(frozen=True, eq=False)
class FenceViewAsync(Instruction):
    space: str

    @staticmethod
    def create(scope: str) -> FenceViewAsync:
        assert scope in ('shared', 'global'), f"Invalid scope for async fence view: {scope}. Supported candidates are 'shared' and 'global'."
        return FenceViewAsync(output=None, inputs=(), space=scope)
