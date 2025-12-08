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

from hidet.ir.expr import Expr, Var
from hidet.ir.primitives.cuda.wgmma import WgmmaConfig, wgmma_async, wgmma_commit_group, wgmma_fence, wgmma_wait_group

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import Tcgen05SwizzleMode
from tilus.extensions.hidet.ir.primitives.cuda.wgmma import wgmma_encode_smem_descriptor
from tilus.ir.inst import Instruction
from tilus.ir.instructions.cuda.wgmma import (
    WgmmaCommitGroupInst,
    WgmmaFenceInst,
    WgmmaMmaSSInst,
    WgmmaWaitGroupInst,
)
from tilus.ir.layout.cuda.tcgen05.smem import canonicalize_shared_layout
from tilus.ir.layout.utils.cute import CuteLayout
from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.target import nvgpu_sm90a


def encode_swizzle_mode(swizzle_mode: Tcgen05SwizzleMode) -> int:
    swizzle_mode_map = {
        Tcgen05SwizzleMode.NO_SWIZZLE: 0,
        Tcgen05SwizzleMode.B32_SWIZZLE: 3,
        Tcgen05SwizzleMode.B64_SWIZZLE: 2,
        Tcgen05SwizzleMode.B128_SWIZZLE: 1,
    }
    return swizzle_mode_map[swizzle_mode]


@dataclass
class SharedMatrixDescriptor:
    addr: Expr | int
    lbo: int
    sbo: int
    base_offset: int
    swizzle_mode: int

    def encoded(self) -> Expr:
        return wgmma_encode_smem_descriptor(
            self.addr >> 4,
            self.lbo >> 4,
            self.sbo >> 4,
            self.base_offset,
            self.swizzle_mode,
        )

    @staticmethod
    def decode(encoded: int) -> SharedMatrixDescriptor:
        return SharedMatrixDescriptor(
            addr=(encoded & 0x3FFF) << 4,
            lbo=((encoded >> 16) & 0x3FFF) << 4,
            sbo=((encoded >> 32) & 0x3FFF) << 4,
            base_offset=(encoded >> 49) & 0x7,
            swizzle_mode=(encoded >> 62) & 0x3,
        )


class WgmmaBaseEmitter(BaseInstEmitter):
    def check_warp_group(self) -> None:
        begin = self.current_thread_group_begin
        end = self.current_thread_group_end
        if begin % 128 != 0 or end - begin != 128:
            raise ValueError("The number of threads in the current thread group must be 128")

    def emit(self, inst: Instruction) -> None:
        self.check_warp_group()
        self.emit_wgmma(inst)

    def emit_wgmma(self, inst: Instruction) -> None:
        raise NotImplementedError("Subclasses must implement this method")


@register_emitter(WgmmaFenceInst, target=nvgpu_sm90a)
class WgmmaFenceEmitter(WgmmaBaseEmitter):
    def emit_wgmma(self, inst: WgmmaFenceInst) -> None:
        self.append(wgmma_fence())


@register_emitter(WgmmaCommitGroupInst, target=nvgpu_sm90a)
class WgmmaCommitGroupEmitter(WgmmaBaseEmitter):
    def emit_wgmma(self, inst: WgmmaCommitGroupInst) -> None:
        self.append(wgmma_commit_group())


@register_emitter(WgmmaWaitGroupInst, target=nvgpu_sm90a)
class WgmmaWaitGroupEmitter(WgmmaBaseEmitter):
    def emit_wgmma(self, inst: WgmmaWaitGroupInst) -> None:
        self.append(wgmma_wait_group(inst.n))


@register_emitter(WgmmaMmaSSInst, target=nvgpu_sm90a)
class WgmmaMmaSSEmitter(WgmmaBaseEmitter):
    def emit_wgmma(self, inst: WgmmaMmaSSInst) -> None:
        a, b, d = inst.inputs
        a_tensor: SharedTensor = a.as_shared_tensor()
        b_tensor: SharedTensor = b.as_shared_tensor()
        d_tensor: RegisterTensor = d.as_register_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        if len(a_shape) != 2 or len(b_shape) != 2 or len(d_shape) != 2:
            raise ValueError(f"MMA requires 2D tensors, but got shapes: a={a_shape}, b={b_shape}, d={d_shape}")
        if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] != d_shape[1]:
            raise ValueError(f"Incompatible shapes for MMA: a={a_shape}, b={b_shape}, d={d_shape}")
        m, n, k = d_shape[0], d_shape[1], a_shape[1]

        a_dtype = a_tensor.dtype
        b_dtype = b_tensor.dtype
        d_dtype = d_tensor.dtype

        inst_m, inst_n, inst_k = inst.get_inst_mnk(m, n, k, a_dtype, b_dtype, d_dtype)
        wgmma_config = WgmmaConfig.get(
            inst_m, inst_n, inst_k, a_dtype.short_name, b_dtype.short_name, d_dtype.short_name
        )

        repeat_m = m // inst_m
        repeat_n = n // inst_n
        repeat_k = k // inst_k

        a_canonical = canonicalize_shared_layout(a_tensor.layout, dtype=a_dtype)
        b_canonical = canonicalize_shared_layout(b_tensor.layout.transpose(), dtype=b_dtype)

        if a_canonical is None:
            raise ValueError(f"Cannot canonicalize the layout of a tensor: {a_tensor.layout}.")
        if b_canonical is None:
            raise ValueError(f"Cannot canonicalize the layout of b tensor: {b_tensor.layout}.")

        a_cute_layout: CuteLayout = a_canonical.swizzled_cute_layout.layout
        b_cute_layout: CuteLayout = b_canonical.swizzled_cute_layout.layout

        a_shared_addr: Var = self.shared_tensor_shared_space_addr[a_tensor]
        b_shared_addr: Var = self.shared_tensor_shared_space_addr[b_tensor]
        d_register_addr: Var = ~(self.tensor2var[d_tensor][0])

        d_local_stride = d_tensor.layout.local_size // repeat_m // repeat_n

        for k in range(repeat_k):
            for i in range(repeat_m):
                for j in range(repeat_n):
                    a_offset = a_cute_layout(i * inst_m, k * inst_k)
                    b_offset = b_cute_layout(j * inst_n, k * inst_k)
                    a_desc = SharedMatrixDescriptor(
                        addr=a_shared_addr + a_offset * a_tensor.dtype.nbytes,
                        lbo=a_canonical.LBO * a_tensor.dtype.nbytes,
                        sbo=a_canonical.SBO * a_tensor.dtype.nbytes,
                        base_offset=0,
                        swizzle_mode=encode_swizzle_mode(a_canonical.swizzle_mode),
                    )
                    b_desc = SharedMatrixDescriptor(
                        addr=b_shared_addr + b_offset * b_tensor.dtype.nbytes,
                        lbo=b_canonical.LBO * b_tensor.dtype.nbytes,
                        sbo=b_canonical.SBO * b_tensor.dtype.nbytes,
                        base_offset=0,
                        swizzle_mode=encode_swizzle_mode(b_canonical.swizzle_mode),
                    )
                    d_offset = (i * repeat_n + j) * d_local_stride
                    self.append(
                        wgmma_async(
                            wgmma_config,
                            a_desc.encoded(),
                            d_register_addr + d_offset,
                            b_desc.encoded(),
                            trans_a=0,  # type: ignore
                            trans_b=0,  # type: ignore
                        )
                    )
