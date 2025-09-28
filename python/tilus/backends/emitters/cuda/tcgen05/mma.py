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


from hidet.ir.type import DataType
from hidet.ir.dtypes import float32, tfloat32, float16, bfloat16, int8, uint8, int32
from tilus.extensions.hidet.ir.dtypes import float8_e4m3, float8_e5m2, float6_e2m3, float4_e2m1
from tilus.backends.codegen import BaseInstEmitter, CodeGenerationFailed, register_emitter
from tilus.ir.layout.cuda.tcgen05.smem import canonicalize_shared_layout
from tilus.extensions.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05CommitMulticastKind,
    Tcgen05CtaGroupKind,
    tcgen05_mma_with_shared_a,
    tcgen05_mma_with_tmem_a,
    Tcgen05MmaKind,
)
from tilus.ir.instructions.cuda.tcgen05 import (
    Tcgen05MmaInst,
)
from tilus.target import nvgpu_sm100
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.utils import gcd


@register_emitter(Tcgen05MmaInst, target=nvgpu_sm100)
class TMemoryMmaEmitter(BaseInstEmitter):
    @staticmethod
    def get_mma_kind(a_dtype: DataType, b_dtype: DataType, d_dtype: DataType) -> Tcgen05MmaKind:
        """
        See Also: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
        """
        if a_dtype == b_dtype == tfloat32 and d_dtype == float32:
            return Tcgen05MmaKind.TF32
        elif all(dtype in (float16, bfloat16) for dtype in (a_dtype, b_dtype)) and d_dtype in (float16, float32):
            return Tcgen05MmaKind.F16
        elif all(dtype in (float8_e4m3, float8_e5m2, float6_e2m3, float4_e2m1) for dtype in (a_dtype, b_dtype)) and d_dtype in (float16, float32):
            return Tcgen05MmaKind.F8F6F4
        elif all(dtype in (int8, uint8) for dtype in (a_dtype, b_dtype)) and d_dtype == int32:
            return Tcgen05MmaKind.I8
        else:
            raise CodeGenerationFailed(f"Cannot infer the MMA kind from the given data types of a, b, and d: {a_dtype}, {b_dtype}, and {d_dtype}.")
    
    @staticmethod
    def get_inst_mnk(mma_kind: Tcgen05MmaKind, m_size: int, n_size: int, k_size: int) -> tuple[int, int, int]:
        """
        See Also: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes
        """
        if mma_kind == Tcgen05MmaKind.F16:
            if m_size not in (64, 128):
                raise CodeGenerationFailed(f"The given m_size is not supported for F16 MMA kind: {m_size}")
            if n_size % 8 != 0:
                raise CodeGenerationFailed(f"The given n_size is not supported for F16 MMA kind: {n_size}")
            inst_m = m_size
            inst_n = gcd(n_size, 256)
            inst_k = 16
            return inst_m, inst_n, inst_k
        else:
            raise NotImplementedError(f"The given MMA kind is not supported yet: {mma_kind}")

    def emit(self, inst: Tcgen05MmaInst) -> None:
        a_tensor: SharedTensor | TMemoryTensor = inst.inputs[0].as_shared_or_tmemory_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        assert len(a_shape) == len(b_shape) == len(d_shape) == 2
        assert a_shape[0] == d_shape[0] and a_shape[1] == b_shape[0] and b_shape[1] == d_shape[1]
        m_size, n_size, k_size = a_shape[0], b_shape[1], a_shape[1]
        if m_size != 128:
            raise NotImplementedError("Only support m_size = 128 for now.")

        mma_kind = self.get_mma_kind(a_tensor.dtype, b_tensor.dtype, d_tensor.dtype)
        inst_m, inst_n, inst_k = self.get_inst_mnk(mma_kind, m_size, n_size, k_size)

        if isinstance(a_tensor, SharedTensor):
            a_canonical = canonicalize_shared_layout(a_tensor.layout, dtype=a_tensor.dtype)
            if a_canonical is None:
                raise CodeGenerationFailed(f"Cannot canonicalize the layout of a tensor: {a_tensor.layout}.")
        else:
            pass
