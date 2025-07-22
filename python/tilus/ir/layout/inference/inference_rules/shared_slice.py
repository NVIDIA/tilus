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
from tilus.ir.instructions import SharedSliceInst
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.shared_layout import shared_compose, shared_row_major


@register_rule(SharedSliceInst)
class SharedSliceRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: SharedSliceInst) -> dict[SharedTensor, SharedLayout]:
        a = inst.shared_input
        b = inst.shared_output
        if a.optional_layout is not None and b.optional_layout is not None:
            return {}
        elif a.optional_layout is not None:
            return {b: a.layout.slice(offsets=inst.offsets, slice_dims=inst.dims, slice_shape=b.shape)}
        elif b.optional_layout is not None:
            b_layout = b.layout.unsqueeze(dims=range(len(a.shape) - len(b.shape)))
            outer_shape = []
            for i in range(len(a.shape)):
                outer_shape.append(a.shape[i] // b_layout.shape[i])
            return {a: shared_compose(shared_row_major(*outer_shape), b_layout).simplify()}
        else:
            return {}
