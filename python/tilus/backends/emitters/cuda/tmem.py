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

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions.cuda.tcgen05 import TMemorySliceInst, TMemoryViewInst
from tilus.target import nvgpu_sm100

# @register_emitter(Tcgen05RelinquishAllocPermitInst, target=nvgpu_sm100)
# class Tcgen05RelinquishAllocPermitEmitter(BaseInstEmitter):
#     def emit(self, inst: Tcgen05RelinquishAllocPermitInst) -> None:
#         self.append(tcgen05_relinquish_alloc_permit(inst.cta_group))


@register_emitter(TMemorySliceInst, target=nvgpu_sm100)
class TMemorySliceEmitter(BaseInstEmitter):
    def emit(self, inst: TMemorySliceInst) -> None:
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        output_tmem_tensor = inst.output.as_tmemory_tensor()
        tmem_addr = self.get_or_allocate_var(tmem_tensor)

        lane_stride = 0x00010000
        column_stride = 0x00000001

        sliced_addr = self.get_or_allocate_var(output_tmem_tensor, name="tmem_slice")
        self.assign(
            sliced_addr,
            tmem_addr + inst.offsets[0] * lane_stride + inst.offsets[1] * column_stride * tmem_tensor.dtype.nbits // 32,
        )


@register_emitter(TMemoryViewInst, target=nvgpu_sm100)
class TMemoryViewEmitter(BaseInstEmitter):
    def emit(self, inst: TMemoryViewInst) -> None:
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        output_tmem_tensor = inst.output.as_tmemory_tensor()

        if (
            tmem_tensor.dtype.nbits * tmem_tensor.shape[1]
            != output_tmem_tensor.dtype.nbits * output_tmem_tensor.shape[1]
        ):
            raise ValueError("The total number of bits must be the same as the original tensor.")

        tmem_addr = self.get_or_allocate_var(tmem_tensor)
        view_addr = self.get_or_allocate_var(output_tmem_tensor, name="tmem_view")
        self.assign(view_addr, tmem_addr)
