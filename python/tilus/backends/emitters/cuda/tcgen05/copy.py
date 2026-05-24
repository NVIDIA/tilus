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


from __future__ import annotations

from dataclasses import dataclass

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.backends.emitters.cuda.tcgen05.smem_desc import SharedMatrixDescriptor
from tilus.hidet.ir.dtypes import uint64
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.cuda.tcgen05 import (
    COLUMN_STRIDE,
    LANE_STRIDE,
    Tcgen05CopyMulticastKind,
    Tcgen05CopyShapeKind,
    Tcgen05CtaGroupKind,
    tcgen05_copy,
)
from tilus.hidet.ir.utils.index_transform import index_deserialize
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05CopyInst
from tilus.ir.layout.cuda.tcgen05.smem import CanonicalSharedLayout, Tcgen05SwizzleMode, canonicalize_shared_layout
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.target import nvgpu_sm100


class GenerationFailedError(Exception):
    pass


@dataclass
class Tcgen05CopyInstMeta:
    shape_kind: Tcgen05CopyShapeKind
    multicast: Tcgen05CopyMulticastKind
    cta_group: Tcgen05CtaGroupKind
    tmem_offset: int
    shared_descriptor: SharedMatrixDescriptor

    def __str__(self) -> str:
        items = []
        for key, value in self.__dict__.items():
            items.append(f"{key}: {value}")
        return "Tcgen05CopyInstMeta(" + ",\n  ".join(items) + "\n)"


_MULTICAST_NAME_TO_KIND: dict[str, Tcgen05CopyMulticastKind] = {
    "": Tcgen05CopyMulticastKind.NONE,
    "warpx4": Tcgen05CopyMulticastKind.WARP_X4,
    "warpx2_02_13": Tcgen05CopyMulticastKind.WARP_X2_02_13,
    "warpx2_01_23": Tcgen05CopyMulticastKind.WARP_X2_01_23,
}

# Per-multicast: required source-row count in SMEM and the candidate shape kinds.
# Smaller shape kinds (R32x128B / R64x128B) are only valid with multicast — the
# table maps each multicast to the shape kind PTX requires for it.
_MULTICAST_TO_SOURCE_ROWS: dict[Tcgen05CopyMulticastKind, int] = {
    Tcgen05CopyMulticastKind.NONE: 128,
    Tcgen05CopyMulticastKind.WARP_X4: 32,
    Tcgen05CopyMulticastKind.WARP_X2_02_13: 64,
    Tcgen05CopyMulticastKind.WARP_X2_01_23: 64,
}

_MULTICAST_TO_SHAPE_KINDS: dict[Tcgen05CopyMulticastKind, tuple[Tcgen05CopyShapeKind, ...]] = {
    Tcgen05CopyMulticastKind.NONE: (
        Tcgen05CopyShapeKind.R128x256B,
        Tcgen05CopyShapeKind.R128x128B,
    ),
    Tcgen05CopyMulticastKind.WARP_X4: (Tcgen05CopyShapeKind.R32x128B,),
    Tcgen05CopyMulticastKind.WARP_X2_02_13: (Tcgen05CopyShapeKind.R64x128B,),
    Tcgen05CopyMulticastKind.WARP_X2_01_23: (Tcgen05CopyShapeKind.R64x128B,),
}


@register_emitter(Tcgen05CopyInst, target=nvgpu_sm100)
class Tcgen05CopyEmitter(BaseInstEmitter):
    def split_canonical_layout(
        self,
        smem_addr: Expr,
        canonical: CanonicalSharedLayout,
        shape_kind: Tcgen05CopyShapeKind,
        multicast: Tcgen05CopyMulticastKind = Tcgen05CopyMulticastKind.NONE,
    ) -> list[Tcgen05CopyInstMeta]:
        """
        Split the canonical shared layout into multiple sub-tensors that can be copied by tcgen05.copy instructions.

        A shared memory tensor might be very large that we need to split it into multiple sub-tensors and
        each sub-tensor is copied by a tcgen05.copy instruction. The smem_addr in returned SharedMatrixDescriptor
        is the offset of the sub-tensor relative to the shared memory tensor in bytes.

        The definition of the canonical layout in Tilus is similar to above table, but it's different since we want to represent the layouts
        in a more natural and extensible way for larger tensors. See the docstring of CanonicalSharedLayout for more details.

        Returns
        -------
        ret: Optional[list[tuple[int, SharedMatrixDescriptor]]]
            The list of instructions, each instruction contains the tmem_offset and shared matrix descriptor for each sub-tensor.
        """
        cute_layout = canonical.swizzled_cute_layout.layout
        m, n = cute_layout.flattened_shape
        assert isinstance(m, int) and isinstance(n, int), "Only static shape is supported in tcgen05.copy emitter"

        if shape_kind.n % canonical.dtype_nbits != 0:
            raise GenerationFailedError(
                "The number of columns in the shape kind must be divisible by the number of bits in the data type"
            )

        inst_m, inst_n = shape_kind.m, shape_kind.n // canonical.dtype_nbits

        if m % inst_m != 0 or n % inst_n != 0:
            raise GenerationFailedError(
                "The number of rows or columns in the shape kind must be divisible by the number of rows or columns in the canonical layout"
            )
        if canonical.major_kind == "MN" and (inst_m % (canonical.T * canonical.S) != 0 or inst_n % 8 != 0):
            raise GenerationFailedError(
                "The number of rows or columns in the shape kind must be divisible by the number of rows or columns in the canonical layout"
            )
        if canonical.major_kind == "K" and (inst_m % 8 != 0 or inst_n % canonical.T != 0):
            raise GenerationFailedError(
                "The number of rows or columns in the shape kind must be divisible by the number of rows or columns in the canonical layout"
            )

        num_m, num_n = m // inst_m, n // inst_n
        nbytes = canonical.dtype_nbits // 8

        instructions: list[Tcgen05CopyInstMeta] = []
        for i in range(num_m):
            for j in range(num_n):
                tmem_offset = i * inst_m * LANE_STRIDE + j * inst_n * COLUMN_STRIDE
                if canonical.major_kind == "MN":
                    if canonical.swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
                        smem_offset = (
                            i * inst_m // (canonical.T * canonical.S) * canonical.SBO + j * inst_n // 8 * canonical.LBO
                        ) * nbytes
                    else:
                        smem_offset = (
                            i * inst_m // (canonical.T * canonical.S) * canonical.LBO + j * inst_n // 8 * canonical.SBO
                        ) * nbytes
                    s_desc = SharedMatrixDescriptor(
                        addr=smem_addr + smem_offset,
                        lbo=canonical.LBO * nbytes,
                        sbo=canonical.SBO * nbytes,
                        base_offset=0,
                        stride_mode=0,
                        swizzle_mode=canonical.swizzle_mode.encode(),
                    )
                elif canonical.major_kind == "K":
                    if canonical.swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
                        smem_offset = (
                            i * inst_m // 8 * canonical.SBO + j * inst_n // (canonical.T * canonical.S) * canonical.LBO
                        ) * nbytes
                        lbo = canonical.LBO * nbytes
                    else:
                        # j0, j1, j2 for shape (T, S, k)
                        _, j1, j2 = index_deserialize(
                            j * inst_n,
                            (canonical.T, canonical.S, canonical.k // (canonical.T * canonical.S)),
                            ranks=[2, 1, 0],
                        )
                        smem_offset = (i * inst_m // 8 * canonical.SBO + j1 * canonical.T + j2 * canonical.LBO) * nbytes
                        lbo = 1 << 4  # assume lbo be 16 so that lbo >> 4 == 1, as required by the documentation
                    s_desc = SharedMatrixDescriptor(
                        addr=smem_addr + smem_offset,
                        lbo=lbo,
                        sbo=canonical.SBO * nbytes,
                        base_offset=0,
                        stride_mode=0,
                        swizzle_mode=canonical.swizzle_mode.encode(),
                    )

                instructions.append(
                    Tcgen05CopyInstMeta(
                        shape_kind=shape_kind,
                        multicast=multicast,
                        cta_group=Tcgen05CtaGroupKind.CTA_1,
                        tmem_offset=tmem_offset,
                        shared_descriptor=s_desc,
                    )
                )

        return instructions

    def generate_instructions(
        self,
        tmem_tensor: TMemoryTensor,
        shared_tensor: SharedTensor,
        multicast: Tcgen05CopyMulticastKind = Tcgen05CopyMulticastKind.NONE,
    ) -> list[Tcgen05CopyInstMeta]:
        dtype = shared_tensor.dtype
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

        for shape_kind in _MULTICAST_TO_SHAPE_KINDS[multicast]:
            try:
                return self.split_canonical_layout(smem_addr, canonical_layout, shape_kind, multicast)
            except GenerationFailedError:
                continue

        raise ValueError("No valid instructions generated")

    def emit(self, inst: Tcgen05CopyInst) -> None:
        shared_tensor = inst.inputs[1].as_shared_tensor()
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()

        self.assert_is_warp_aligned(inst, "tcgen05.copy is a warp-cooperative instruction")

        multicast_kind = _MULTICAST_NAME_TO_KIND.get(inst.multicast)
        if multicast_kind is None:
            raise ValueError("Unknown multicast {!r} on Tcgen05CopyInst".format(inst.multicast))

        if len(shared_tensor.shape) != 2:
            raise ValueError("The shared tensor must be a 2D tensor, got shape {}".format(shared_tensor.shape))
        if len(tmem_tensor.shape) != 2:
            raise ValueError("The tensor memory tensor must be a 2D tensor, got shape {}".format(tmem_tensor.shape))
        expected_rows = _MULTICAST_TO_SOURCE_ROWS[multicast_kind]
        if shared_tensor.shape[0] != expected_rows:
            raise ValueError(
                "tcgen05.copy multicast={!r} requires the shared tensor to have {} rows, got {}".format(
                    inst.multicast, expected_rows, shared_tensor.shape[0]
                )
            )
        if tmem_tensor.layout.lane_offset != 0:
            raise NotImplementedError("The first lane of the tmem tensor must be 0")

        tmem_base_addr = self.tensor2var[tmem_tensor]

        with self.single_thread():
            insts = self.generate_instructions(tmem_tensor, shared_tensor, multicast_kind)
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
                        predicate=self.contexts.leader_lane_ctx.leader_lane,
                    )
                )
