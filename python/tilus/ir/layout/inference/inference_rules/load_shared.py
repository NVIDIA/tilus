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
from tilus import RegisterLayout, SharedLayout
from tilus.ir import RegisterTensor, SharedTensor
from tilus.ir.analyzers.grid_analyzer import analyze_grid
from tilus.ir.instructions import LoadMatrixInst, LoadSharedGenericInst, LoadSharedInst
from tilus.ir.instructions.cuda.ldmatrix import LoadMatrixConfig
from tilus.ir.layout import LayoutOperationError, ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.ops import shared_row_major_swizzle
from tilus.utils import gcd


@register_rule(LoadMatrixInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
class LoadSharedInferSwizzledSharedRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadSharedInst | LoadSharedGenericInst | LoadMatrixInst
    ) -> dict[SharedTensor, SharedLayout]:
        a = inst.shared_input
        b = inst.register_output

        if not (a.optional_layout is None and b.optional_layout is not None):
            return {}

        for config in LoadMatrixConfig.all():
            if config.nbytes != a.dtype.nbytes:
                continue
            try:
                ops.divide(b.layout, config.ldmatrix_layout)
            except LayoutOperationError:
                continue

            # use swizzle layout since we are using ldmatrix instruction

            return {a: shared_row_major_swizzle(dtype_nbytes=a.dtype.nbytes, shape=a.shape)}

        return {}


@register_rule(LoadMatrixInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
class LoadSharedInferRowMajorSharedRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadSharedInst | LoadSharedGenericInst | LoadMatrixInst
    ) -> dict[SharedTensor, SharedLayout]:
        a = inst.shared_input
        b = inst.register_output

        if not (a.optional_layout is None and b.optional_layout is not None):
            return {}

        from tilus.ir.layout.ops.shared_ops import shared_row_major

        return {a: shared_row_major(*a.shape)}


@register_rule(LoadMatrixInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
class LoadSharedInferRegisterRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadSharedInst | LoadSharedGenericInst | LoadMatrixInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        shared = inst.shared_input
        register = inst.register_output

        if not (shared.has_layout() and not register.has_layout()):
            return {}

        axes, offset = shared.layout.as_axes_mapping()

        info = analyze_grid(
            shape=shared.shape,
            axes=axes,
            expr=offset,
            analysis=ctx.analysis,
        )

        for dim in range(len(shared.shape)):
            factor = gcd(
                info[dim].divisibility,
                info[dim].continuity,
                128 // shared.dtype.nbits,
                shared.shape[dim],
            )
            if factor > 1:
                lhs_shape = list(shared.shape)
                lhs_shape[dim] = shared.shape[dim] // factor
                rhs_shape = [1 if i != dim else factor for i in range(len(shared.shape))]
                layout = ops.auto_local_spatial(num_threads=ctx.num_threads, shape=lhs_shape) * ops.local(*rhs_shape)
                return {register: layout}

        return {register: ops.auto_local_spatial(num_threads=ctx.num_threads, shape=shared.shape)}
