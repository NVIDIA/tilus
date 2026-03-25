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
# ruff: noqa: I001  (import order matters — func/math must come before cuda)
from .func import register_primitive_function, is_primitive_function, lookup_primitive_function

# base primitive functions — must be loaded before cuda (cuda's @initialize needs primitives.pow etc.)
# pylint: disable=redefined-builtin
from .math import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, expm1, abs
from .math import max, min, exp, pow, sqrt, rsqrt, erf, ceil, log, log2, log10, log1p, round, floor, trunc
from .math import isfinite, isinf, isnan, make_vector, atan2, mod

# debug primitives
from .debug import printf, __builtin_assume

# cuda primitive functions and variables
from . import cuda
from .cuda import threadIdx, blockIdx
from .cuda import syncthreads, syncwarp, lds128, sts128, shfl_sync, shfl_up_sync, shfl_down_sync, shfl_xor_sync
from .cuda import ldg256, ldg128, ldg64, ldg32, ldg256_lu, ldg128_lu, ldg64_lu, ldg32_lu, ldg16
from .cuda import stg512, stg256, stg128, stg64, stg32, stg16
from .cuda import lds64, lds32, lds16, lds8
from .cuda import sts64, sts32, sts16, sts8
from .cuda import active_mask, set_kernel_max_dynamic_smem_bytes
from .cuda import cvt, cvtv
from .cuda import tcgen05_alloc, tcgen05_dealloc, tcgen05_relinquish_alloc_permit
from .cuda import tcgen05_ld, tcgen05_st, tcgen05_wait

# extension modules (use @initialize/@script, must come after base primitives)
from . import swizzle, utils

cuda._load_extension_modules()
