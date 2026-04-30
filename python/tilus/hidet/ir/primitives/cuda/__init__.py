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
# ruff: noqa: I001  (import order matters — base primitives first, extensions after)

# Base hidet primitives (don't use @script, safe to import early)
from . import math
from . import mma

from .barrier import (
    barrier_arrive,
    barrier_sync,
    cp_async_barrier_arrive,
    mbarrier_arrive,
    mbarrier_arrive_and_expect_tx,
    mbarrier_complete_transaction,
    mbarrier_expect_transaction,
    mbarrier_init,
    mbarrier_invalidate,
    mbarrier_test_wait,
    mbarrier_try_wait,
    mbarrier_wait,
)
from .cluster import this_cluster
from .cp_async import cp_async, cp_async_commit_group, cp_async_wait_all, cp_async_wait_group
from .cvt import cvt, cvtv
from .errchk import check_cuda_error
from .half import fma_f16x2, sub_f16x2
from .ldst import (
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
)
from .lop3 import lop3
from .memcpy import memcpy_async
from .prmt import prmt
from .shfl import active_mask, shfl_down_sync, shfl_sync, shfl_up_sync, shfl_xor_sync
from .smem import set_kernel_max_dynamic_smem_bytes
from .sync import bar_sync, bar_sync_aligned, bar_warp_sync, syncthreads, syncwarp
from .tcgen05_cp import make_tcgen05_cp_desc, matrix_descriptor_encode, tcgen05_cp, tcgen05_shift
from .tcgen05_ldst import tcgen05_ld, tcgen05_st, tcgen05_wait
from .tensor_map import create_tensor_map
from .tmem import (
    compute_tmem_address,
    compute_tmem_offset_address,
    get_register_count,
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_relinquish_alloc_permit,
)
from .vars import blockDim, blockIdx, gridDim, threadIdx


def _load_extension_modules():
    """Load extension primitive modules that use @initialize/@script.

    These must be loaded AFTER the base primitives module is fully initialized,
    because the @script decorator's transpiler references primitives.pow etc.
    """
    from . import (  # noqa: F811
        bfloat16,
        cast,
        clc,
        control,
        copy_async_bulk,
        copy_async_tensor,
        elect,
        fence,
        float32,
        integer_intrinsics,
        mapa,
        mbarrier,
        subbyte,
        tcgen05,
    )
