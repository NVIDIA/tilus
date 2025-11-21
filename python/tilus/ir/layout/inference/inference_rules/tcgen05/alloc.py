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
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05AllocInst
from tilus.ir.layout import TMemoryLayout
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.layout.ops.tmemory_ops import tmemory_row_major
from tilus.ir.tensor import TMemoryTensor


@register_rule(Tcgen05AllocInst)
class Tcgen05AllocRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05AllocInst) -> dict[TMemoryTensor, TMemoryLayout]:
        tmem = inst.tmemory_output
        if tmem.optional_layout is not None:
            return {}
        return {tmem: tmemory_row_major(tmem.shape)}
