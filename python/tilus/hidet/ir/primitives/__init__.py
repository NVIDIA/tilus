# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# cpu primitive functions
# cuda primitive functions and variables
# Extension modules
from . import cuda, swizzle, utils
from .cuda import (
    active_mask,
    blockIdx,
    cvt,
    cvtv,
    ldg16,
    ldg32,
    ldg32_lu,
    ldg64,
    ldg64_lu,
    ldg128,
    ldg128_lu,
    ldg256,
    ldg256_lu,
    lds8,
    lds16,
    lds32,
    lds64,
    lds128,
    set_kernel_max_dynamic_smem_bytes,
    shfl_down_sync,
    shfl_sync,
    shfl_up_sync,
    shfl_xor_sync,
    stg16,
    stg32,
    stg64,
    stg128,
    stg256,
    stg512,
    sts8,
    sts16,
    sts32,
    sts64,
    sts128,
    syncthreads,
    syncwarp,
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_relinquish_alloc_permit,
    tcgen05_st,
    tcgen05_wait,
    threadIdx,
)

# function used to debug
from .debug import __builtin_assume, printf
from .func import is_primitive_function, lookup_primitive_function, register_primitive_function

# base primitive functions
# pylint: disable=redefined-builtin
from .math import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    expm1,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    log1p,
    log2,
    log10,
    make_vector,
    max,
    min,
    mod,
    pow,
    round,
    rsqrt,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
)
