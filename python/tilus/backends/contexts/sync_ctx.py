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

from hidet.ir.expr import Expr

from tilus.backends.context import BaseEmitContext
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import mbarrier_sync_shared


class SyncContext(BaseEmitContext):
    def __post_init__(self):
        # map of (start_thread, end_thread) to the barrier (uint32 address in shared memory)
        # used for synchronization among a subset of threads in the thread block
        self.thread_group_barrier: dict[tuple[int, int], Expr] = {}

    def sync(self):
        thread_begin = self.codegen.thread_group_stack.thread_begin[-1]
        thread_end = self.codegen.thread_group_stack.thread_end[-1]
        key = (thread_begin, thread_end)
        if key not in self.thread_group_barrier:
            # allocate a new barrier for this thread group
            self.thread_group_barrier[key] = self.contexts.barrier_alloc_ctx.allocate_barriers(
                counts=[thread_end - thread_begin]
            )[0]
        barrier_addr = self.thread_group_barrier[key]
        self.kernel_append(mbarrier_sync_shared(barrier_addr))
