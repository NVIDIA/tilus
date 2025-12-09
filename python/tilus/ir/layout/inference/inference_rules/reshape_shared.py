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
from tilus import SharedLayout
from tilus.ir import SharedTensor
from tilus.ir.instructions import ReshapeSharedInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule


@register_rule(ReshapeSharedInst)
class ReshapeSharedRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: ReshapeSharedInst) -> dict[SharedTensor, SharedLayout]:
        src: SharedTensor = inst.shared_input
        dst: SharedTensor = inst.shared_output
        if src.optional_layout is None and dst.optional_layout is None:
            return {}
        elif src.optional_layout is None:
            return {src: ops.shared_reshape(dst.layout, src.shape)}
        elif dst.optional_layout is None:
            return {dst: ops.shared_reshape(src.layout, dst.shape)}
        else:
            return {}
