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
import functools

from hidet.ir import dtypes
from hidet.ir.dtypes import DataType


@functools.cache
def dtype_from_torch(torch_dtype):
    import torch

    if torch_dtype is None:
        return None

    if isinstance(torch_dtype, DataType):
        return torch_dtype

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    mapping = {
        torch.float64: dtypes.float64,
        torch.float32: dtypes.float32,
        torch.float: dtypes.float32,
        torch.bfloat16: dtypes.bfloat16,
        torch.float8_e4m3fn: dtypes.float8_e4m3,
        torch.float8_e5m2: dtypes.float8_e5m2,
        torch.float16: dtypes.float16,
        torch.half: dtypes.float16,
        torch.int64: dtypes.int64,
        torch.int32: dtypes.int32,
        torch.int16: dtypes.int16,
        torch.int8: dtypes.int8,
        torch.uint8: dtypes.uint8,
        torch.uint16: dtypes.uint16,
        torch.uint32: dtypes.uint32,
        torch.uint64: dtypes.uint64,
        torch.bool: dtypes.boolean,
        torch.double: dtypes.float64,
        torch.complex64: dtypes.complex64,
        torch.complex128: dtypes.complex128,
    }
    return mapping[torch_dtype]


@functools.cache
def dtype_to_torch(dtype):
    import torch

    mapping = {
        dtypes.float64: torch.float64,
        dtypes.float32: torch.float32,
        dtypes.bfloat16: torch.bfloat16,
        dtypes.float8_e4m3: torch.float8_e4m3fn,
        dtypes.float8_e5m2: torch.float8_e5m2,
        dtypes.float16: torch.float16,
        dtypes.int64: torch.int64,
        dtypes.int32: torch.int32,
        dtypes.int16: torch.int16,
        dtypes.int8: torch.int8,
        dtypes.uint8: torch.uint8,
        dtypes.uint16: torch.uint16,
        dtypes.uint32: torch.uint32,
        dtypes.uint64: torch.uint64,
        dtypes.boolean: torch.bool,
    }
    return mapping[dtype]
