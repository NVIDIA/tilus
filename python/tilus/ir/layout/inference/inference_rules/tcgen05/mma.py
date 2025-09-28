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
from operator import is_
from hidet.ir.dtypes import float8_e4m3, float8_e5m2
from hidet.ir.type import DataType
from hidet.ir.dtypes import float32, tfloat32, float16, bfloat16, int8, uint8, int32
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05MmaInst
from tilus.extensions.hidet.ir.dtypes import float6_e2m3, float6_e3m2, float4_e2m1
from tilus.ir.layout.cuda.tcgen05.smem import (
    Tcgen05SwizzleMode,
    generate_canonical_layout,
    get_shared_layout_from_canonical,
)
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceError,
    LayoutInferenceRule,
    register_rule,
)
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import Tcgen05MmaKind
from tilus.ir.tensor import SharedTensor, TMemoryTensor, Tensor
from tilus.ir.layout import SharedLayout


@register_rule(Tcgen05MmaInst)
class Tcgen05MmaRule(LayoutInferenceRule):

    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05MmaInst) -> dict[SharedTensor, SharedLayout]:
        a_tensor: SharedTensor | TMemoryTensor = inst.inputs[0].as_shared_or_tmemory_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        if not len(a_shape) == len(b_shape) == len(d_shape) == 2:
            raise LayoutInferenceError(f"A, B, and D must have 2 dimensions, but got {len(a_shape)}, {len(b_shape)}, and {len(d_shape)}.")
        if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] != d_shape[1]:
            raise LayoutInferenceError(f"A, B, and D must have compatible shapes, but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}.")
        m, n, k = d_shape[0], d_shape[1], a_shape[1]

        ret = {}
        if isinstance(a_tensor, SharedTensor) and not a_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    a_layout = generate_canonical_layout(
                        shape=(m, k),
                        dtype=a_tensor.dtype,
                        major_kind="K",
                        swizzle_mode=swizzle_mode
                    ).as_shared_layout()
                    ret[a_tensor] = a_layout
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
                    b_layout = generate_canonical_layout(
                        shape=(n, k),
                        dtype=b_tensor.dtype,
                        major_kind="K",
                        swizzle_mode=swizzle_mode
                    ).as_shared_layout().transpose()
                    ret[b_tensor] = b_layout
                except ValueError:
                    continue
                else:
                    break
        
        return ret
