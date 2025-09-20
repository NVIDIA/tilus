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
from tilus.ir.instructions.cuda.tmem import Tcgen05CopyInst
from tilus.ir.layout.cuda.tcgen05_smem import (
    Tcgen05SwizzleMode,
    generate_canonical_layout,
    get_shared_layout_from_canonical,
)
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.tensor import SharedLayout, SharedTensor


@register_rule(Tcgen05CopyInst)
class TMemoryCopyRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05CopyInst) -> dict[SharedTensor, SharedLayout]:
        src = inst.inputs[1].as_shared_tensor()

        if src.has_layout():
            return {}

        # by default, we use the non-swizzled canonical layout
        assert len(src.shape) == 2
        canonical_layout = generate_canonical_layout(
            (src.shape[0], src.shape[1]), src.dtype, "K", Tcgen05SwizzleMode.NO_SWIZZLE
        )
        shared_layout = get_shared_layout_from_canonical(canonical_layout)
        return {src: shared_layout}
