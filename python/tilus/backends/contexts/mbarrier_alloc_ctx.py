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

from typing import Sequence

from hidet.ir.builders import StmtBuilder
from hidet.ir.dtypes import uint32, uint64
from hidet.ir.expr import Expr, Var
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.primitives.cuda.vars import threadIdx

from tilus.backends.context import BaseEmitContext
from tilus.extensions.hidet.ir.primitives.cuda.fence import fence_mbarrier_init_cluster
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import mbarrier_init_shared
from tilus.ir.layout import ops
from tilus.ir.tensor import SharedTensor


class BarrierAllocContext(BaseEmitContext):
    """Context used to manage the allocation of barriers."""

    def __post_init__(self):
        self.counts: list[Expr] = []
        self.barriers: list[Var] = []
        self.barrier_addr: Var = Var("barriers", type=uint32)

    def finalize(self):
        # allocate shared memory for all barriers
        num_barriers = len(self.counts)

        if num_barriers == 0:
            # No barriers to allocate
            return

        tensor = SharedTensor(dtype=uint64, shape=(num_barriers,), optional_layout=ops.shared_row_major(num_barriers))
        virtual_smem_addr = self.contexts.smem_alloc_ctx.allocate_shared_tensor(tensor, nbytes=tensor.nbytes)
        sb = StmtBuilder()
        sb.declare(
            v=self.barrier_addr,
            init=cvta_generic_to_shared(dynamic_shared_memory(byte_offset=virtual_smem_addr, dtype=uint64)),
        )

        for i in range(num_barriers):
            sb.declare(v=self.barriers[i], init=self.barrier_addr + uint32(i * uint64.nbytes))

        with sb.if_then(threadIdx.x == 0):  # type: ignore
            for i in range(num_barriers):
                sb.append(mbarrier_init_shared(mbarrier_addr=self.barriers[i], arrive_count=uint32(self.counts[i])))
        sb.append(fence_mbarrier_init_cluster())
        sb.append(syncthreads())
        self.kernel_prepend(sb.finish())

    def allocate_barriers(self, counts: Sequence[Expr | int]) -> list[Var]:
        """
        Allocate a list of barriers with given counts.

        Each barrier is a 64-bit data structure stored in shared memory.
        This function returns the address of the first barrier in the shared space.
        """
        barrier_vars = [Var("barrier_{}".format(c), type=uint32) for c in counts]
        self.counts.extend([uint32(c) if isinstance(c, int) else c for c in counts])
        self.barriers.extend(barrier_vars)
        return barrier_vars
