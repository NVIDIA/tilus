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
from .cuda.semaphore import LockSemaphoreInst, ReleaseSemaphoreInst
from .cuda.simt_dot import SimtDotInst
from .cuda.mma_dot import DotInst
from .cuda.ldmatrix import LoadMatrixInst, LoadMatrixConfig
from .cuda.cp_async import (
    CopyAsyncCommitGroupInst,
    CopyAsyncGenericInst,
    CopyAsyncInst,
    CopyAsyncWaitAllInst,
    CopyAsyncWaitGroupInst,
)
from .cuda.mbarrier import InitBarrierInst, ArriveBarrierInst, ArriveRemoteBarrierInst, WaitBarrierInst
from .cuda.bulk_cp_async import BulkCopyAsyncSharedToGlobalInst, BulkCopyAsyncSharedToClusterSharedInst, BulkCopyAsyncGlobalToSharedInst, BulkCopyAsyncGlobalToClusterSharedInst
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
    GlobalSliceInst,
    GlobalViewInst,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedGenericInst,
    LoadSharedInst,
    ModInst,
    MulInst,
    PrintTensorInst,
    ReduceInst,
    RepeatInst,
    RepeatInterleaveInst,
    SharedSliceInst,
    ShuffleDownInst,
    ShuffleUpInst,
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
