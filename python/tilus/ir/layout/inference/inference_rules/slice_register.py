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
from typing import Optional, Sequence

from tilus import RegisterLayout
from tilus.ir.instructions import SliceAssignInst, SliceRegisterInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


class BaseSliceRegisterRule(LayoutInferenceRule):
    @staticmethod
    def get_full_layout(num_threads: int, shape: Sequence[int]) -> RegisterLayout:
        lhs = ops.reduce(ops.spatial(num_threads), dims=[0], keepdims=True)
        rhs = ops.local(*shape)
        full_layout = lhs * rhs
        return full_layout

    @staticmethod
    def get_sliced_layout(full_layout: RegisterLayout, slice_dims: Optional[Sequence[int]]) -> RegisterLayout:
        if slice_dims is None:
            slice_dims = range(len(full_layout.shape))
        reduce_dims = [i for i in range(len(full_layout.shape)) if i not in slice_dims]
        sliced_layout = ops.reduce(full_layout, dims=reduce_dims, keepdims=False)
        return sliced_layout


@register_rule(SliceRegisterInst)
class SliceRegisterRule(BaseSliceRegisterRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: SliceRegisterInst) -> dict[RegisterTensor, RegisterLayout]:
        src = inst.register_input
        dst = inst.register_output
        ret = {}

        if src.optional_layout is None:
            src_layout = SliceRegisterRule.get_full_layout(ctx.num_threads, src.shape)
            ret[src] = src_layout
        else:
            src_layout = src.optional_layout

        if dst.optional_layout is None:
            dst_layout = SliceRegisterRule.get_sliced_layout(src_layout, inst.dims)
            ret[dst] = dst_layout

        return ret


@register_rule(SliceAssignInst)
class SliceAssignRule(BaseSliceRegisterRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: SliceAssignInst) -> dict[RegisterTensor, RegisterLayout]:
        dst = inst.inputs[0].as_register_tensor()
        src = inst.inputs[1].as_register_tensor()
        ret = {}

        if dst.optional_layout is None:
            dst_layout = SliceRegisterRule.get_full_layout(ctx.num_threads, dst.shape)
            ret[dst] = dst_layout
        else:
            dst_layout = dst.optional_layout

        if src.optional_layout is None:
            src_layout = SliceRegisterRule.get_sliced_layout(dst_layout, inst.dims)
            ret[src] = src_layout

        return ret
