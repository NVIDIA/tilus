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
# ruff: noqa: I001  (do not sort imports, we need to import the extensions first)
from . import extensions as _

from hidet.ir.dtypes import (
    bf16,
    bfloat16,
    boolean,
    f16,
    f32,
    f64,
    float16,
    float32,
    float64,
    i8,
    i16,
    i32,
    i64,
    int8,
    int16,
    int32,
    int64,
    tfloat32,
    u8,
    u16,
    u32,
    u64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from hidet.ir.type import void_p

from tilus.extensions.hidet.ir.dtypes import (
    f3e1m1,
    f4e2m1,
    f5e2m2,
    f5e3m1,
    f6e2m3,
    f6e3m2,
    f6e4m1,
    f7e2m4,
    f7e3m3,
    f7e4m2,
    f7e5m1,
    f8e4m3,
    f8e5m2,
    float3_e1m1,
    float4_e2m1,
    float5_e2m2,
    float5_e3m1,
    float6_e2m3,
    float6_e3m2,
    float6_e4m1,
    float7_e2m4,
    float7_e3m3,
    float7_e4m2,
    float7_e5m1,
    float8_e4m3,
    float8_e5m2,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    int1b,
    int2b,
    int3b,
    int4b,
    int5b,
    int6b,
    int7b,
    u1,
    u2,
    u3,
    u4,
    u5,
    u6,
    u7,
    uint1b,
    uint2b,
    uint3b,
    uint4b,
    uint5b,
    uint6b,
    uint7b,
)
from hidet.ir.type import DataType
from tilus.ir.layout import RegisterLayout, SharedLayout
from tilus.ir.tensor import RegisterTensor, SharedTensor, GlobalTensor
from tilus.lang.instantiated_script import InstantiatedScript
from tilus.lang.script import Script, autotune
from tilus.lang.constructs.state import Class
from tilus.lang.classes import Pipeline
from tilus.lang.constructs.structs import Dim3
from tilus.tensor import empty, from_torch, full, ones, rand, randint, randn, view_torch, zeros

from . import kernels, logging, option, utils, testing
from .target import get_current_target
from .version import __version__
