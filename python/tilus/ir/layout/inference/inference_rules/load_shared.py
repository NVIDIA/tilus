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
from tilus import RegisterLayout, SharedLayout
from tilus.ir import RegisterTensor, SharedTensor
from tilus.ir.analyzers.grid_analyzer import analyze_grid
from tilus.ir.instructions import LoadSharedInst
from tilus.ir.layout import LayoutOperationError, ops
from tilus.ir.layout.cuda.ldmatrix import LoadMatrixConfig
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.ops import shared_row_major_swizzle
from tilus.utils import gcd


@register_rule(LoadSharedInst)
class LoadSharedInferSwizzledSharedRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: LoadSharedInst) -> dict[SharedTensor, SharedLayout]:
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


@register_rule(LoadSharedInst)
class LoadSharedInferRowMajorSharedRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: LoadSharedInst) -> dict[SharedTensor, SharedLayout]:
        a = inst.shared_input
        b = inst.register_output

        if not (a.optional_layout is None and b.optional_layout is not None):
            return {}

        from tilus.ir.layout.ops.shared_ops import shared_row_major

        return {a: shared_row_major(*a.shape)}


@register_rule(LoadSharedInst)
class LoadSharedInferLdmatrixRegisterRule(LayoutInferenceRule):
    """Infer an ldmatrix-compatible register layout when the shared layout supports it.

    The layout is: auto_local_spatial(num_warps, lhs_shape) * ldmatrix_layout,
    where num_warps = num_threads // 32 and lhs_shape tiles the ldmatrix atom across the tensor.
    """

    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: LoadSharedInst) -> dict[RegisterTensor, RegisterLayout]:
        shared = inst.shared_input
        register = inst.register_output

        if not (shared.has_layout() and not register.has_layout()):
            return {}

        if len(register.shape) != 2 or ctx.num_threads % 32 != 0:
            return {}

        dtype = register.dtype
        num_warps = ctx.num_threads // 32

        for config in LoadMatrixConfig.all():
            if dtype.nbytes != config.nbytes:
                continue

            ldmatrix_layout = config.ldmatrix_layout

            # check shape divisibility by the ldmatrix tile
            if any(s % ls != 0 for s, ls in zip(register.shape, ldmatrix_layout.shape)):
                continue

            # check the outer tiling fits the warp count
            lhs_shape = [s // ls for s, ls in zip(register.shape, ldmatrix_layout.shape)]
            if lhs_shape[0] * lhs_shape[1] < num_warps:
                continue

            layout = ops.auto_local_spatial(num_threads=num_warps, shape=lhs_shape) * ldmatrix_layout
            return {register: layout}

        return {}


@register_rule(LoadSharedInst)
class LoadSharedInferRegisterRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: LoadSharedInst) -> dict[RegisterTensor, RegisterLayout]:
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
