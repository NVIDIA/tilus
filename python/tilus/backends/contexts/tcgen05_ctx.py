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

from typing import Optional

from tilus.backends.codegen import BaseEmitContext, FunctionCodegen, register_emit_context


@register_emit_context
class Tcgen05EmitContext(BaseEmitContext):
    _current: Optional[Tcgen05EmitContext] = None

    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)

        # the cta_group used by the tcgen05 instructions
        # PTX requires all tcgen05 instructions in the kernel use the same cta_group value
        # thus, we only ask the user to give cta_group in the tcgen05.alloc instruction and we track it here
        # for other tcgen05 instructions to use
        self.cta_group: Optional[int] = None

    @staticmethod
    def current() -> Tcgen05EmitContext:
        if Tcgen05EmitContext._current is None:
            raise RuntimeError("No active Tcgen05EmitContext found.")
        return Tcgen05EmitContext._current

    def set_cta_group(self, cta_group: int) -> None:
        assert cta_group in (1, 2)
        if self.cta_group is None:
            self.cta_group = cta_group
        elif self.cta_group != cta_group:
            raise ValueError(
                f"All tcgen05 instructions in the kernel must use the same cta_group value, "
                f"but got {self.cta_group} and {cta_group}"
            )

    def get_cta_group(self) -> int:
        if self.cta_group is None:
            raise ValueError(
                "cta_group is not set yet. Please ensure that a tcgen05.alloc instruction is emitted "
                "before any other tcgen05 instructions."
            )
        return self.cta_group
