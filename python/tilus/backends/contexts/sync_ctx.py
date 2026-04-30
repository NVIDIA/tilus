# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from tilus.backends.context import BaseEmitContext
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.cuda.mbarrier import mbarrier_sync_shared
from tilus.hidet.ir.primitives.cuda.sync import bar_sync_aligned, bar_warp_sync, syncthreads, syncwarp


class SyncContext(BaseEmitContext):
    # Named barriers 1..4 are available for inter-warp sync within a CTA.
    # Barrier 0 is reserved for __syncthreads().
    MAX_NAMED_BARRIERS = 4

    def __post_init__(self):
        # map of (thread_begin, thread_end) -> barrier_id (1..4) for named barrier sync
        self.named_barrier_map: dict[tuple[int, int], int] = {}
        # next barrier id to allocate (starts at 1, max 4)
        self.next_barrier_id: int = 1

        # map of (thread_begin, thread_end) -> mbarrier shared memory address (fallback)
        self.thread_group_barrier: dict[tuple[int, int], Expr] = {}

    def _try_allocate_named_barrier(self, thread_begin: int, thread_end: int) -> Optional[int]:
        """Try to allocate a named barrier for the given thread range.

        Returns the barrier id (1..4) if successful, None if all slots are used.
        """
        key = (thread_begin, thread_end)
        if key in self.named_barrier_map:
            return self.named_barrier_map[key]
        if self.next_barrier_id > self.MAX_NAMED_BARRIERS:
            return None
        barrier_id = self.next_barrier_id
        self.next_barrier_id += 1
        self.named_barrier_map[key] = barrier_id
        return barrier_id

    def sync(self) -> Optional[Expr]:
        thread_begin = self.codegen.thread_group_stack.thread_begin[-1]
        thread_end = self.codegen.thread_group_stack.thread_end[-1]
        current_threads = thread_end - thread_begin
        total_threads = self.codegen.thread_group_stack.thread_end[0] - self.codegen.thread_group_stack.thread_begin[0]

        if thread_begin == 0 and thread_end == total_threads:
            # full CTA sync
            return syncthreads()
        elif current_threads == 1:
            # single thread, no sync needed
            return None
        elif thread_begin % 32 == 0 and current_threads == 32:
            # exactly one warp
            return syncwarp()
        elif thread_begin % 32 != 0:
            # not warp-aligned start: must use mbarrier
            return self._mbarrier_sync(thread_begin, thread_end)
        elif current_threads < 32:
            # intra-warp subset (warp-aligned start, fewer than 32 threads)
            # compute membermask: bits [0..current_threads) within the warp
            membermask = (1 << current_threads) - 1
            return bar_warp_sync(membermask)
        elif current_threads % 32 != 0:
            # multi-warp but not a multiple of 32: must use mbarrier
            return self._mbarrier_sync(thread_begin, thread_end)
        else:
            # multi-warp, warp-aligned, multiple of 32: try named barrier
            barrier_id = self._try_allocate_named_barrier(thread_begin, thread_end)
            if barrier_id is not None:
                return bar_sync_aligned(barrier_id=barrier_id, thread_count=current_threads)
            else:
                return self._mbarrier_sync(thread_begin, thread_end)

    def _mbarrier_sync(self, thread_begin: int, thread_end: int) -> Expr:
        """Fallback: use mbarrier for partial thread group synchronization."""
        key = (thread_begin, thread_end)
        if key not in self.thread_group_barrier:
            _, barrier_vars = self.contexts.barrier_alloc_ctx.allocate_barriers(counts=[thread_end - thread_begin])
            self.thread_group_barrier[key] = barrier_vars[0]
        barrier_addr = self.thread_group_barrier[key]
        return mbarrier_sync_shared(barrier_addr)
