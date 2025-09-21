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


from __future__ import annotations

from dataclasses import dataclass

from hidet.ir.dtypes import uint64
from hidet.ir.expr import Expr

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.backends.emitters.cuda.tcgen05.allocation import COLUMN_STRIDE
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05CopyMulticastKind,
    Tcgen05CopyShapeKind,
    Tcgen05CtaGroupKind,
    tcgen05_copy,
    tcgen05_encode_smem_descriptor,
)
from tilus.ir.instructions.cuda.tmem import Tcgen05CopyInst
from tilus.ir.layout.cuda.tcgen05_smem import CanonicalSharedLayout, canonicalize_shared_layout
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.target import nvgpu_sm100


@dataclass
class SharedMatrixDescriptor:
    start_addr: Expr | int
    lbo: int
    sbo: int
    base_offset: int
    stride_mode: int
    swizzle_mode: int

    def encoded(self) -> Expr:
        return tcgen05_encode_smem_descriptor(
            self.start_addr,
            self.lbo,
            self.sbo,
            self.base_offset,
            self.stride_mode,
            self.swizzle_mode,
        )

    @staticmethod
    def decode(encoded: int) -> SharedMatrixDescriptor:
        return SharedMatrixDescriptor(
            start_addr=encoded & 0x3FFF,
            lbo=(encoded >> 16) & 0x3FFF,
            sbo=(encoded >> 32) & 0x3FFF,
            base_offset=(encoded >> 49) & 0x7,
            stride_mode=(encoded >> 52) & 0x1,
            swizzle_mode=(encoded >> 61) & 0x7,
        )


@dataclass
class Tcgen05CopyInstMeta:
    shape_kind: Tcgen05CopyShapeKind
    multicast: Tcgen05CopyMulticastKind
    cta_group: Tcgen05CtaGroupKind
    tmem_offset: int
    shared_descriptor: SharedMatrixDescriptor


@register_emitter(Tcgen05CopyInst, target=nvgpu_sm100)
class Tcgen05CopyEmitter(BaseInstEmitter):
    def generate_instructions(
        self, tmem_tensor: TMemoryTensor, shared_tensor: SharedTensor
    ) -> list[Tcgen05CopyInstMeta]:
        dtype = shared_tensor.dtype
        shape = shared_tensor.shape
        canonical_layout: CanonicalSharedLayout | None = canonicalize_shared_layout(
            shared_tensor.layout, tmem_tensor.dtype
        )
        if canonical_layout is None:
            msg = [
                "The following <dtype, shared_layout> cannot be canonicalized:",
                f"  dtype: {dtype.name}",
                f"  shared_layout: {shared_tensor.layout}",
            ]
            raise ValueError("\n".join(msg))
        smem_addr = self.shared_tensor_shared_space_addr[shared_tensor]
        ret = []
        for shape_kind in [
            Tcgen05CopyShapeKind.R128x256B,
            Tcgen05CopyShapeKind.R128x128B,
        ]:
            column_bits = shape_kind.as_int_tuple()[1]
            assert column_bits % dtype.nbits == 0
            column_elements = column_bits // dtype.nbits
            if shape[1] % column_elements != 0:
                continue
            if shape[0] != 128:
                continue
            num_inst_columns = shape[1] // column_elements
            for inst_column in range(num_inst_columns):
                tmem_offset = inst_column * (column_bits // 32 * COLUMN_STRIDE)
                smem_offset = inst_column * (
                    column_elements // canonical_layout.atom_shape[1] * canonical_layout.atom_strides[1] * dtype.nbytes
                )

                shared_descriptor = SharedMatrixDescriptor(
                    start_addr=(smem_addr + smem_offset) >> 4,
                    lbo=(canonical_layout.LBO * dtype.nbytes) >> 4,
                    sbo=(canonical_layout.SBO * dtype.nbytes) >> 4,
                    base_offset=0,
                    stride_mode=0,  # 0 for relative mode and 1 for absolute mode
                    swizzle_mode=canonical_layout.swizzle_mode.encode(),
                )

                inst_meta = Tcgen05CopyInstMeta(
                    shape_kind=shape_kind,
                    multicast=Tcgen05CopyMulticastKind.NONE,
                    cta_group=Tcgen05CtaGroupKind.CTA_1,
                    tmem_offset=tmem_offset,
                    shared_descriptor=shared_descriptor,
                )
                ret.append(inst_meta)
            break
        else:
            raise ValueError("No valid instructions generated")
        return ret

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
            insts = self.generate_instructions(tmem_tensor, shared_tensor)
            for inst_meta in insts:
                s_desc = self.declare_var("s_desc", tp=uint64, init=inst_meta.shared_descriptor.encoded())
                t_addr = tmem_base_addr + inst_meta.tmem_offset

                self.append(
                    tcgen05_copy(
                        taddr=t_addr,
                        sdesc=s_desc,
                        cta_group=inst_meta.cta_group,
                        shape=inst_meta.shape_kind,
                        multicast=inst_meta.multicast,
                    )
                )
