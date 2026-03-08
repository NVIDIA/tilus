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
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05MmaSSInst, Tcgen05MmaTSInst
from tilus.ir.layout import SharedLayout
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
from tilus.ir.tensor import SharedTensor, TMemoryTensor


@register_rule(Tcgen05MmaSSInst)
class Tcgen05MmaSSRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05MmaSSInst) -> dict[SharedTensor, SharedLayout]:
        a_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        if not len(a_shape) == len(b_shape) == len(d_shape) == 2:
            raise LayoutInferenceError(
                f"A, B, and D must have 2 dimensions, but got {len(a_shape)}, {len(b_shape)}, and {len(d_shape)}."
            )

        if inst.cta_group == 1:
            if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] != d_shape[1]:
                raise LayoutInferenceError(
                    f"A, B, and D must have compatible shapes, but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}."
                )
        elif inst.cta_group == 2:
            if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] * 2 != d_shape[1]:
                raise LayoutInferenceError(
                    f"For cta_group=2, A, B, and D must have compatible shapes with B's shape[1] being half of D's shape[1], but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}."
                )
        else:
            raise LayoutInferenceError(f"cta_group must be 1 or 2, but got {inst.cta_group}.")

        ret = {}
        if not a_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    m, k = a_tensor.shape
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
                    k, n = b_tensor.shape
                    b_layout_canonical = generate_canonical_layout(
                        shape=(n, k), dtype=b_tensor.dtype, major_kind="K", swizzle_mode=swizzle_mode
                    )
                    ret[b_tensor] = b_layout_canonical.as_shared_layout().permute(dims=[1, 0])
                except ValueError:
                    continue
                else:
                    break

        return ret


@register_rule(Tcgen05MmaTSInst)
class Tcgen05MmaTSRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05MmaTSInst) -> dict[SharedTensor, SharedLayout]:
        a_tensor: TMemoryTensor = inst.inputs[0].as_tmemory_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        if not len(a_shape) == len(b_shape) == len(d_shape) == 2:
            raise LayoutInferenceError(
                f"A, B, and D must have 2 dimensions, but got {len(a_shape)}, {len(b_shape)}, and {len(d_shape)}."
            )
        if inst.cta_group == 1:
            if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] != d_shape[1]:
                raise LayoutInferenceError(
                    f"A, B, and D must have compatible shapes, but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}."
                )
        elif inst.cta_group == 2:
            if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] * 2 != d_shape[1]:
                raise LayoutInferenceError(
                    f"For cta_group=2, A, B, and D must have compatible shapes with B's shape[1] being half of D's shape[1], but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}."
                )
        else:
            raise LayoutInferenceError(f"cta_group must be 1 or 2, but got {inst.cta_group}.")

        ret = {}
        if not b_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    k, n = b_tensor.shape
                    b_layout = (
                        generate_canonical_layout(
                            shape=(n, k), dtype=b_tensor.dtype, major_kind="K", swizzle_mode=swizzle_mode
                        )
                        .as_shared_layout()
                        .permute(dims=[1, 0])
                    )
                    ret[b_tensor] = b_layout
                except ValueError:
                    continue
                else:
                    break

        return ret
