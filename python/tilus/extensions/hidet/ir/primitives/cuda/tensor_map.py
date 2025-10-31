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

from enum import Enum

from hidet.ir.expr import Expr
from hidet.ir.stmt import BlackBoxStmt, Stmt
from hidet.ir.type import DataType, OpaqueType, PointerType

CUtensorMapType = OpaqueType("CUtensorMap")
CUTensorMapPointerType = PointerType(CUtensorMapType)


class TensorMapDataType(Enum):
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    INT32 = "INT32"
    UINT64 = "UINT64"
    INT64 = "INT64"
    FLOAT16 = "FLOAT16"
    FLOAT32 = "FLOAT32"
    FLOAT64 = "FLOAT64"
    BFLOAT16 = "BFLOAT16"
    FLOAT32_FTZ = "FLOAT32_FTZ"
    TFLOAT32 = "TFLOAT32"
    TFLOAT32_FTZ = "TFLOAT32_FTZ"
    UINT4x16_ALIGN8B = "16U4_ALIGN8B"
    UINT4x16_ALIGN16B = "16U4_ALIGN16B"
    UINT6x16_ALIGN16B = "16U6_ALIGN16B"

    def cpp_str(self):
        return f"CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_{self.value}"

    @classmethod
    def from_dtype(cls, dtype: DataType) -> TensorMapDataType:
        if dtype.nbits == 8:
            return cls.UINT8
        else:
            return cls(dtype.name.upper())


class TensorMapInterleave(Enum):
    NONE = "NONE"
    B16 = "16B"
    B32 = "32B"

    def cpp_str(self):
        return f"CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_{self.value}"


class TensorMapL2Promotion(Enum):
    NONE = "NONE"
    B64 = "L2_64B"
    B128 = "L2_128B"
    B256 = "L2_256B"

    def cpp_str(self):
        return f"CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_{self.value}"


class TensorMapSwizzle(Enum):
    NONE = "NONE"
    B32 = "32B"
    B64 = "64B"
    B128 = "128B"
    B128_ATOM_32B = "128B_ATOM_32B"
    B128_ATOM_32B_FLIP_8B = "128B_ATOM_32B_FLIP_8B"
    B128_ATOM_64B = "128B_ATOM_64B"

    def cpp_str(self):
        return f"CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_{self.value}"


class TensorMapFloatOOBFill(Enum):
    NONE = "NONE"
    NAN_REQUEST_ZERO_FMA = "NAN_REQUEST_ZERO_FMA"

    def cpp_str(self):
        return f"CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_{self.value}"


def encode_tensor_map(
    tensor_map: Expr,
    dtype: TensorMapDataType,
    rank: Expr,
    tensor_ptr: Expr,
    shape: Expr,
    strides: Expr,
    box_shape: Expr,
    elem_strides: Expr,
    interleave: TensorMapInterleave,
    swizzle: TensorMapSwizzle,
    l2_promotion: TensorMapL2Promotion,
    oob_fill: TensorMapFloatOOBFill,
) -> Stmt:
    """Encode a tensor map.

    Parameters
    ----------
    tensor_map: Expr
        The address of the tensor map to store the encoded result in host memory.
    dtype: TensorMapDataType
        The data type of the tensor.
    rank: Expr
        The rank of the tensor. It should be of int32 type.
    tensor_ptr: Expr
        The pointer to the tensor data in global memory. It should be of a pointer type.
    shape: Expr
        The shape of the tensor. It should be of uint64[rank] type.
    strides: Expr
        The strides of the tensor. It should be of uint64[rank-1] type or a pointer to uint64 in host memory. The stride
         for the first dimension in the tensor shape is always 1 and is not included in the strides array. For example,
         for a tensor of shape [2, 3, 4], the strides array should be [2, 6] instead of [1, 2, 6].
    box_shape: Expr
        The box shape of the tensor. It should be of uint32[rank] type or a pointer to uint32 in host memory.
    elem_strides: Expr
        The element strides of the tensor. It should be of uint32[rank] type or a pointer to uint32 in host memory.
    interleave: TensorMapInterleave
        The interleave mode.
    swizzle: TensorMapSwizzle
        The swizzle mode.
    l2_promotion: TensorMapL2Promotion
        The L2 promotion mode.
    oob_fill: TensorMapFloatOOBFill
        The out-of-bounds fill mode.

    Returns
    -------
    ret: Stmt
        The statement that encodes the tensor map.
    """
    template_string = (
        """cuTensorMapEncodeTiled(
    {{}},
    {dtype},
    {{}},
    {{}},
    {{}},
    {{}},
    {{}},
    {{}},
    {interleave},
    {swizzle},
    {l2_promotion},
    {oob_fill}
);
""".format(
            dtype=dtype.cpp_str(),
            interleave=interleave.cpp_str(),
            swizzle=swizzle.cpp_str(),
            l2_promotion=l2_promotion.cpp_str(),
            oob_fill=oob_fill.cpp_str(),
        )
        + r"""{{cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err) << "\n";}}"""
    )

    return BlackBoxStmt(template_string, tensor_map, rank, tensor_ptr, shape, strides, box_shape, elem_strides)
