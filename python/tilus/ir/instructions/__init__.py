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
from tilus.ir.inst import Instruction

from .annotation import AnnotateLayoutInst
from .cuda.cluster_sync import ClusterSyncThreadsInst
from .cuda.cp_async import (
    CopyAsyncCommitGroupInst,
    CopyAsyncGenericInst,
    CopyAsyncInst,
    CopyAsyncWaitAllInst,
    CopyAsyncWaitGroupInst,
)
from .cuda.cp_async_bulk import (
    CopyAsyncBulkGlobalToClusterSharedInst,
    CopyAsyncBulkGlobalToSharedInst,
    CopyAsyncBulkSharedToClusterSharedInst,
    CopyAsyncBulkSharedToGlobalInst,
)
from .cuda.ldmatrix import LoadMatrixConfig, LoadMatrixInst
from .cuda.mbarrier import AllocBarrierInst, ArriveBarrierInst, WaitBarrierInst
from .cuda.mma_dot import DotInst
from .cuda.semaphore import LockSemaphoreInst, ReleaseSemaphoreInst
from .cuda.simt_dot import SimtDotInst
from .generic import (
    AddInst,
    AllocateGlobalInst,
    AllocateRegisterInst,
    AllocateSharedInst,
    AssignInst,
    CastInst,
    DivInst,
    ElementwiseBinaryBaseInst,
    ElementwiseBinaryInst,
    ElementwiseUnaryBaseInst,
    ElementwiseUnaryInst,
    ExitInst,
    FormatPrintInst,
    FreeSharedInst,
    GlobalViewInst,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedGenericInst,
    LoadSharedInst,
    ModInst,
    MulInst,
    PermuteSharedInst,
    PrintTensorInst,
    ReduceInst,
    RepeatInst,
    RepeatInterleaveInst,
    ShuffleDownInst,
    ShuffleUpInst,
    SliceAssignInst,
    SliceGlobalInst,
    SliceRegisterInst,
    SliceSharedInst,
    SqueezeInst,
    StoreGlobalGenericInst,
    StoreGlobalInst,
    StoreSharedGenericInst,
    StoreSharedInst,
    SubInst,
    SyncReduceThreadsInst,
    SyncThreadsInst,
    TransposeInst,
    UnsqueezeInst,
    ViewInst,
    WhereInst,
)
