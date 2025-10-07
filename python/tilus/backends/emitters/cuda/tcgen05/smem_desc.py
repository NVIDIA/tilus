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
from tilus.backends.emitters.cuda.tcgen05.allocation import COLUMN_STRIDE, LANE_STRIDE
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05CopyMulticastKind,
    Tcgen05CopyShapeKind,
    Tcgen05CtaGroupKind,
    tcgen05_copy,
    tcgen05_encode_smem_descriptor,
)
from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05CopyInst
from tilus.ir.layout.cuda.tcgen05.smem import CanonicalSharedLayout, Tcgen05SwizzleMode, canonicalize_shared_layout
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.target import nvgpu_sm100

@dataclass
class SharedMatrixDescriptor:
    """
        Each tcgen05.copy instruction copies a sub-tensor with the following layout:
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        | Major-ness     | Swizzling mode           | Canonical Layout without swizzling   | Swizzling on the previous column    |
        +================+==========================+======================================+=====================================+
        | MN-major       | No-swizzling or          | ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)) | Swizzle<0, 4, 3>                    |
        |                | Interleaved              |                                      |                                     |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        |                | 32B Swizzling            | ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO)) | Swizzle<1, 4, 3>                    |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        |                | 64B Swizzling            | ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO)) | Swizzle<2, 4, 3>                    |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        |                | 128B Swizzling           | ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO)) | Swizzle<3, 4, 3>                    |
        +================+==========================+======================================+=====================================+
        | K-major        | No-swizzling or          | ((8,m),(T,2k)):((1T,SBO),(1,LBO))    | Swizzle<0, 4, 3>                    |
        |                | Interleaved              |                                      |                                     |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        |                | 32B Swizzling            | ((8,m),(T,2k)):((2T,SBO),(1,T))      | Swizzle<1, 4, 3>                    |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        |                | 64B Swizzling            | ((8,m),(T,2k)):((4T,SBO),(1,T))      | Swizzle<2, 4, 3>                    |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        |                | 128B Swizzling           | ((8,m),(T,2k)):((8T,SBO),(1,T))      | Swizzle<3, 4, 3>                    |
        +----------------+--------------------------+--------------------------------------+-------------------------------------+
        where
        - T = 128 / sizeof-elements-in-bits T represents scale factor which normalizes matrix element types to 128-bits.
        - m represents the number of repeating patterns across rows.
        - k represents the number of repeating patterns across columns.
        (The table is is from: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-canonical-layouts.)

    """
    addr: Expr | int
    lbo: int
    sbo: int
    base_offset: int
    stride_mode: int
    swizzle_mode: int

    def encoded(self) -> Expr:
        return tcgen05_encode_smem_descriptor(
            self.addr >> 4,
            self.lbo >> 4,
            self.sbo >> 4,
            self.base_offset,
            self.stride_mode,
            self.swizzle_mode,
        )

    @staticmethod
    def decode(encoded: int) -> SharedMatrixDescriptor:
        return SharedMatrixDescriptor(
            addr=(encoded & 0x3FFF) << 4,
            lbo=((encoded >> 16) & 0x3FFF) << 4,
            sbo=((encoded >> 32) & 0x3FFF) << 4,
            base_offset=(encoded >> 49) & 0x7,
            stride_mode=(encoded >> 52) & 0x1,
            swizzle_mode=(encoded >> 61) & 0x7,
        )