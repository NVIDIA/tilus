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
from tilus.hidet.ir.primitives.cuda.atomic import (
    atomic_add,
    atomic_cas,
    atomic_exchange,
    atomic_max,
    atomic_min,
    atomic_sub,
)
from tilus.hidet.ir.primitives.cuda.cluster import this_cluster
from tilus.hidet.ir.primitives.cuda.cp_async import (
    cp_async,
    cp_async_commit_group,
    cp_async_wait_all,
    cp_async_wait_group,
)
from tilus.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from tilus.hidet.ir.primitives.cuda.fastintdiv import fast_intdiv, fast_intmod
from tilus.hidet.ir.primitives.cuda.ldst import load, store
from tilus.hidet.ir.primitives.cuda.memcpy import memcpy, memcpy_async
from tilus.hidet.ir.primitives.cuda.mma import MmaConfig, ldmatrix, mma_sync
from tilus.hidet.ir.primitives.cuda.mutex import (
    acquire_lock,
    acquire_seq_semaphore,
    release_lock,
    release_seq_semaphore,
)
from tilus.hidet.ir.primitives.cuda.setmaxnreg import setmaxnreg
from tilus.hidet.ir.primitives.cuda.shfl import shfl_down_sync, shfl_sync, shfl_up_sync, shfl_xor_sync
from tilus.hidet.ir.primitives.cuda.smem import dynamic_shared_memory, set_kernel_max_dynamic_smem_bytes
from tilus.hidet.ir.primitives.cuda.sync import (
    syncthreads,
    syncthreads_and,
    syncthreads_count,
    syncthreads_or,
    syncwarp,
)
from tilus.hidet.ir.primitives.cuda.time import nano_sleep
from tilus.hidet.ir.primitives.cuda.vars import blockDim, blockIdx, gridDim, threadIdx
from tilus.hidet.ir.primitives.cuda.wgmma import (
    WgmmaConfig,
    make_wgmma_desc,
    wgmma_async,
    wgmma_commit_group,
    wgmma_fence,
    wgmma_wait_group,
)
from tilus.hidet.lang.constructs.declare import register_tensor, shared_tensor

from . import contexts
