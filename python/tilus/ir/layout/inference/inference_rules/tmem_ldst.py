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
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import Tcgen05LoadStoreShapeKind
from tilus.ir.instructions.cuda.tmem import TMemoryLoadInst, TMemoryStoreInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.cuda.tmem_ldst import get_ldst_layout
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceError,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.layout.ops.register_ops import local
from tilus.ir.tensor import RegisterTensor, TMemoryTensor


class TMemoryLdstRule(LayoutInferenceRule):
    @staticmethod
    def get_register_layout(tmem_tensor: TMemoryTensor) -> RegisterLayout:
        dtype = tmem_tensor.dtype
        total_rows = tmem_tensor.shape[0]
        total_columns_bits = tmem_tensor.shape[1] * dtype.nbits

        for shape_kind in [
            Tcgen05LoadStoreShapeKind.R32x32B,
            Tcgen05LoadStoreShapeKind.R16x64B,
            Tcgen05LoadStoreShapeKind.R16x128B,
            Tcgen05LoadStoreShapeKind.R16x256B,
        ]:
            shape_rows = shape_kind.rows()
            shape_columns_bits = shape_kind.columns_bits()
            if total_rows % shape_rows != 0 or total_columns_bits % shape_columns_bits != 0:
                continue
            rows_repeat = total_rows // shape_rows
            columns_repeat = total_columns_bits // shape_columns_bits
            if shape_rows == 16:
                intra_warp_rows_repeat = 2 if rows_repeat % 2 == 0 else 1
            elif shape_rows == 32:
                intra_warp_rows_repeat = 1
            else:
                raise NotImplementedError(f"Unsupported shape: {shape_kind}")
            inter_warp_rows_repeat = rows_repeat // intra_warp_rows_repeat

            assert 32 % tmem_tensor.dtype.nbits == 0

            atom = get_ldst_layout(shape_kind)
            return (
                local(1, columns_repeat).spatial(inter_warp_rows_repeat, 1).local(intra_warp_rows_repeat, 1)
                * atom
                * local(1, 32 // tmem_tensor.dtype.nbits)
            )
        raise NotImplementedError(f"Unsupported tensor to perform tcgen05 load/store: {tmem_tensor}")

    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: TMemoryLoadInst | TMemoryStoreInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        if isinstance(inst, TMemoryLoadInst):
            tmem_tensor = inst.inputs[0].as_tmemory_tensor()
            regs_tensor = inst.register_output
        elif isinstance(inst, TMemoryStoreInst):
            tmem_tensor = inst.inputs[0].as_tmemory_tensor()
            regs_tensor = inst.inputs[1].as_register_tensor()
        else:
            raise NotImplementedError(inst)

        if regs_tensor.has_layout():
            return {}

        # slice the tmem tensor with offsets and shape
        sliced_tmem_tensor = TMemoryTensor.create(
            dtype=tmem_tensor.dtype, shape=regs_tensor.shape, first_lane=tmem_tensor.first_lane + inst.offsets[0]
        )

        # check that the lane matches the threads
        lane_begin = sliced_tmem_tensor.first_lane
        lane_end = lane_begin + sliced_tmem_tensor.shape[0]
        thread_begin = ctx.thread_begin
        thread_end = ctx.thread_end

        if thread_end - thread_begin != lane_end - lane_begin:
            raise LayoutInferenceError(
                f"The lane range {lane_begin} to {lane_end} does not match the thread range {thread_begin} to {thread_end}"
            )
        if thread_begin % 128 != lane_begin:
            raise LayoutInferenceError(
                f"The lane range {lane_begin} to {lane_end} does not match the thread range {thread_begin} to {thread_end}"
            )

        # infer the layout
        layout = TMemoryLdstRule.get_register_layout(sliced_tmem_tensor)

        assert layout.shape == regs_tensor.shape, (
            f"Layout shape {layout.shape} does not match tensor shape {regs_tensor.shape} in {inst}"
        )

        return {regs_tensor: layout}


@register_rule(TMemoryLoadInst)
class TMemoryLoadRule(TMemoryLdstRule):
    pass


@register_rule(TMemoryStoreInst)
class TMemoryStoreRule(TMemoryLdstRule):
    pass
