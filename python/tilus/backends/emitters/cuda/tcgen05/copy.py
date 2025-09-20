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


from dataclasses import dataclass

from hidet.ir.dtypes import uint64
from hidet.ir.expr import Expr

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05CopyMulticastKind,
    Tcgen05CopyShapeKind,
    Tcgen05CtaGroupKind,
    tcgen05_copy,
    tcgen05_encode_smem_descriptor,
)
from tilus.ir.instructions.cuda.tmem import Tcgen05CopyInst
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.target import nvgpu_sm100


@dataclass
class SharedMatrixDescriptor:
    start_addr: Expr
    ldo_or_lda: int
    sdo: int
    base_offset: int
    stride_mode: int
    swizzle_mode: int

    def encoded(self) -> Expr:
        return tcgen05_encode_smem_descriptor(
            self.start_addr,
            self.ldo_or_lda,
            self.sdo,
            self.base_offset,
            self.stride_mode,
            self.swizzle_mode,
        )


@dataclass
class Tcgen05CopyInstMeta:
    shape: Tcgen05CopyShapeKind
    multicast: Tcgen05CopyMulticastKind
    cta_group: Tcgen05CtaGroupKind
    tmem_offset: int
    shared_descriptor: SharedMatrixDescriptor


@register_emitter(Tcgen05CopyInst, target=nvgpu_sm100)
class Tcgen05CopyEmitter(BaseInstEmitter):
    def generate_instructions(
        self, tmem_tensor: TMemoryTensor, shared_tensor: SharedTensor
    ) -> list[Tcgen05CopyInstMeta]:
        # multicast_kind = Tcgen05CopyMulticastKind.NONE
        # cta_group = Tcgen05CtaGroupKind.CTA_1
        # for shape_kind in [
        #     Tcgen05CopyShapeKind.R128x256B,
        #     Tcgen05CopyShapeKind.R128x128B,
        #     # Tcgen05CopyShapeKind.R64x128B,  # todo: support these
        #     # Tcgen05CopyShapeKind.R32x128B,
        #     # Tcgen05CopyShapeKind.R4x128B,
        # ]:
        #     pass
        raise NotImplementedError("Tcgen05CopyEmitter is not implemented")

    def check_warp_group(self) -> None:
        begin = self.current_thread_group_begin
        end = self.current_thread_group_end
        if begin % 128 != 0 or end - begin != 128:
            raise ValueError("The number of threads in the current thread group must be 128")

    def emit(self, inst: Tcgen05CopyInst) -> None:
        shared_tensor = inst.inputs[1].as_shared_tensor()
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()

        self.check_warp_group()

        if len(shared_tensor.shape) != 2:
            raise ValueError("The shared tensor must be a 2D tensor")
        if shared_tensor.shape[0] != 128:
            raise NotImplementedError("The number of rows in the shared tensor must be 128")
        if tmem_tensor.first_lane != 0:
            raise NotImplementedError("The first lane of the tmem tensor must be 0")

        tmem_base_addr = self.tensor2var[tmem_tensor]

        with self.if_then(self.current_thread == 0):
            insts = self.generate_instructions(shared_tensor)
            for inst_meta in insts:
                s_desc = self.declare_var("s_desc", tp=uint64, init=inst_meta.shared_descriptor.encoded())
                t_addr = tmem_base_addr + inst_meta.tmem_offset

                self.append(
                    tcgen05_copy(
                        taddr=t_addr,
                        sdesc=s_desc,
                        cta_group=inst_meta.cta_group,
                        shape=inst_meta.shape,
                        multicast=inst_meta.multicast,
                    )
                )
