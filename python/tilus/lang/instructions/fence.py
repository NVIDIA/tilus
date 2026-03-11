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
import typing
from typing import Literal, Optional, Sequence

from hidet.ir.expr import Expr, as_expr

from tilus.ir.tensor import RegisterTensor

from .root import InstructionGroup


class FenceInstructionGroup(InstructionGroup):
    def async_view(self, scope: str) -> None:
        if scope not in ('shared', 'global'):
            raise ValueError(f"Invalid scope for async fence view: {scope}. Supported candidates are 'shared' and 'global'.")
        self._builder.fence_view_async(scope=scope)
