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
from tilus.ir.instructions.cuda.wgmma import WgmmaMmaSSInst
from tilus.ir.layout import RegisterLayout, SharedLayout
from tilus.ir.layout.cuda.tcgen05.smem import (
    Tcgen05SwizzleMode,
    generate_canonical_layout,
)
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceError,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.layout.ops.register_ops import local
from tilus.ir.tensor import RegisterTensor, SharedTensor


def generate_wgmma_register_layout(m: int, n: int, inst_m: int, inst_n: int, inst_k: int) -> RegisterLayout:
    # See also: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-register-fragment
    T = inst_k // 8
    repeat_m = m // inst_m
    repeat_n = n // inst_n
    return (
        local(repeat_m, repeat_n)
        .column_spatial(inst_m // 16, 1)
        .column_local(2, inst_n // T // 4)
        .spatial(8, 4)
        .local(T)
    )


@register_rule(WgmmaMmaSSInst)
class WgmmaMmaSSRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: WgmmaMmaSSInst
    ) -> dict[SharedTensor | RegisterTensor, SharedLayout | RegisterLayout]:
        a_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: RegisterTensor = inst.inputs[2].as_register_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        if not len(a_shape) == len(b_shape) == len(d_shape) == 2:
            raise LayoutInferenceError(
                f"A, B, and D must have 2 dimensions, but got {len(a_shape)}, {len(b_shape)}, and {len(d_shape)}."
            )
        if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] != d_shape[1]:
            raise LayoutInferenceError(
                f"A, B, and D must have compatible shapes, but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}."
            )
        m, n, k = d_shape[0], d_shape[1], a_shape[1]

        ret: dict[SharedTensor | RegisterTensor, SharedLayout | RegisterLayout] = {}
        if not a_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    a_layout_canonical = generate_canonical_layout(
                        shape=(m, k), dtype=a_tensor.dtype, major_kind="K", swizzle_mode=swizzle_mode
                    )
                    ret[a_tensor] = a_layout_canonical.as_shared_layout()
                except ValueError:
                    continue
                else:
                    break
        if not b_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    b_layout_canonical = generate_canonical_layout(
                        shape=(n, k), dtype=b_tensor.dtype, major_kind="K", swizzle_mode=swizzle_mode
                    )
                    ret[b_tensor] = b_layout_canonical.as_shared_layout().permute(dims=[1, 0])
                except ValueError:
                    continue
                else:
                    break
        if not d_tensor.has_layout():
            inst_m, inst_n, inst_k = WgmmaMmaSSInst.get_inst_mnk(
                m, n, k, a_tensor.dtype, b_tensor.dtype, d_tensor.dtype
            )
            d_layout = generate_wgmma_register_layout(m, n, inst_m, inst_n, inst_k)
            ret[d_tensor] = d_layout
        return ret
