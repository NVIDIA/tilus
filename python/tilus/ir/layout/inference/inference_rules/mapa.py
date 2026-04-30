# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tilus import RegisterLayout
from tilus.ir.instructions.cuda.mapa import MapSharedAddrInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(MapSharedAddrInst)
class MapSharedAddrRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: MapSharedAddrInst) -> dict[RegisterTensor, RegisterLayout]:
        addr = inst.register_input
        out = inst.register_output

        if addr.optional_layout is not None and out.optional_layout is None:
            return {out: addr.layout}
        elif out.optional_layout is not None and addr.optional_layout is None:
            return {addr: out.layout}
        elif addr.optional_layout is None and out.optional_layout is None:
            return {out: ops.replicated(*out.shape, num_workers=ctx.num_threads)}
        return {}
