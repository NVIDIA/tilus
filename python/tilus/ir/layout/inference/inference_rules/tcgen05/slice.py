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
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05SliceInst
from tilus.ir.layout import TMemoryLayout
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.ops.tmemory_ops import tmemory_slice
from tilus.ir.tensor import TMemoryTensor


@register_rule(Tcgen05SliceInst)
class Tcgen05SliceRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05SliceInst) -> dict[TMemoryTensor, TMemoryLayout]:
        tmem = inst.tmemory_output
        if tmem.optional_layout is not None:
            return {}
        assert inst.tmemory_input.optional_layout is not None, "The input tensor must have a layout."
        return {
            tmem: tmemory_slice(
                tmem_layout=inst.tmemory_input.optional_layout,
                lane_offset=int(inst.offsets[-2]),
                slice_dims=inst.slice_dims,
                shape=tmem.shape,
            )
        }
