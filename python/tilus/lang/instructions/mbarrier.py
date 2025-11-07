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
import typing
from typing import Optional, Sequence
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, as_expr
from tilus.ir.tensor import TMemoryTensor, RegisterTensor, SharedTensor
from tilus.ir.builders import StmtBuilder
from tilus.ir.inst import InstructionError
from tilus.utils import is_power_of_two

from .root import InstructionGroup

class BarrierInstructionGroup(InstructionGroup):
    producer_initial_phase = 1
    consumer_initial_phase = 0

    @typing.overload
    def alloc(self, count: Optional[Expr | int] = None) -> Expr:
        """Allocate a barrier in shared memory and get its address in shared space.

         A barrier is an 64-bit data structure, encoded as uint64, in shared memory.
         A barrier contains the following information in the 64 bits:

         - The current phase of the barrier (i.e., phase): 0 or 1.
         - The count of pending arrivals in the current phase: 1 to 2^20 - 1.
         - The count of expected arrivals in the next phase: 0 to 2^20 - 1.
         - The count of pending asynchronous memory transactions in the current phase (i.e., `tx-count`):
           -(2^20 - 1) to 2^20 - 1.

         This instruction allocates an uint64 in shared memory to be used as a mbarrier. The parameter `count` specifies the
         expected arrivals for the barrier.  After initialization, the barrier will have the following initial state:

         - phase = 0
         - pending arrivals = counts[i]
         - expected arrivals = counts[i]
         - tx-count = 0

         When `count` is not provided, it defaults to the number of threads in the current thread group.

         Asynchronous memory copy instructions (e.g., `copy_async_tensor` instructions) that
         take a barrier as an argument will:

         - increase the `tx-count` by the number of bytes to be copied before the copy starts.
         - decrease the `tx-count` by the number of bytes copied after the copy completes asynchronously.

         The `mbarrier.arrive` instruction will decrease the pending arrivals by the number of threads in the thread
         group that call the instruction.

         The `mbarrier.wait` instruction will make the thread group wait until the given phase has finished.

         Once the following conditions are met for the current phase:

         - pending arrivals == 0
         - tx-count == 0

         The barrier will switch to the next phase, and the following will happen:

         - phase = phase ^ 1
         - pending arrivals = expected arrivals in the next phase
         - expected arrivals does not change
         - tx-count = 0

         Parameters
        ----------
         count: Expr | int, optional
             The number of threads that must arrive at the barrier before any of them can proceed. It must be evaluated
             to a positive int32. When not provided, it defaults to the number of threads in the current thread group.

         Returns
         -------
         ret: Expr
             The shared memory address in shared space that points to the allocated barrier. The shared memory address has
             uint32 data type.
        """
        ...

    @typing.overload
    def alloc(self, count: Sequence[Expr | int]) -> RegisterTensor:
        """
        Allocate multiple barriers in shared memory and get their addresses.

        This instruction allocates multiple barriers in shared memory, and returns a register tensor containing the
        shared memory addresses of the allocated barriers. The register tensor has shape (len(count),) and dtype uint32.
        Each barrier is initialized with the corresponding expected arrivals specified in the `count` sequence.

        Parameters
        ----------
        count: Sequence[Expr | int]
            A sequence specifying the number of threads that must arrive at each barrier before any of them can proceed.

        Returns
        -------
        ret: RegisterTensor
            A register tensor of shape (len(count),) and dtype uint32, containing the shared memory addresses of the allocated barriers.

        See Alos
        --------
        See also the single barrier allocation method for more details on barrier behavior.
        """
        ...

    def alloc(self, count: Sequence[Expr | int] | Optional[Expr | int] = None) -> RegisterTensor | Expr:
        counts: list[Expr | None]
        if isinstance(count, Sequence):
            counts = [as_expr(c) if isinstance(c, (Expr, int)) else None for c in count]
        else:
            counts = [as_expr(count) if isinstance(count, (Expr, int)) else None]
        tensor = self._builder.allocate_barrier(counts)
        if isinstance(count, Sequence):
            return tensor
        else:
            return self._builder.tensor_item_value(tensor)

    def arrive(self, barrier: Expr | RegisterTensor, per_thread_count: Expr | int = 1) -> None:
        """Arrive at a barrier.

        This instruction decreases the pending arrivals of given barrier by `per thread count` * `num threads in thread group`.
        Each thread in the current thread group is assumed to arrive with `per thread count`.

        Parameters
        ----------
        barrier: Expr | RegisterTensor
            The uint32 integer representing the address of the barrier in shared space. It can also be a register tensor
            with single element representing the address of the barrier.
        per_thread_count: Expr | int
            The number of arrivals contributed by each thread in the current thread group. It must be evaluated to a positive int32.
            By default, it is 1.
        """
        self._builder.arrive_barrier(barrier, per_thread_count=per_thread_count)

    def multicast_arrive(self, barrier: Expr | RegisterTensor, per_barrier_count: Expr | int = 1) -> None:
        """Arrive the barriers in all thread blocks in the cluster.

        This instruction decreases the pending arrivals of given barrier by `per barrier count`. It also decreases the mbarriers
        at other blocks in the cluster by `per barrier count`.

        Parameters
        ----------
        barrier: Expr | RegisterTensor
            The uint32 integer representing the address of the barrier in shared space. It can also be a register tensor
            with single element representing the address of the barrier.
        per_barrier_count: Expr | int
            The number of arrivals contributed by each barrier in the cluster. It must be evaluated to a positive int32.
            By default, it is 1.
        """
        pass

    def wait(self, barrier: Expr | RegisterTensor, phase: Expr | RegisterTensor | int) -> None:
        """Wait at a barrier.

        This instruction makes the threads in the current thread group wait at the specified barrier until the pending
        arrivals and tx-count of the given phase are both zero.

        When the barrier's current phase is not equal to the specified `phase`, the threads will proceed without waiting
        since the specified phase has already finished.

        When the barrier's current phase is equal to the specified `phase`, the threads will wait until both the pending
        arrivals and tx-count of the current phase are zero. Once these conditions are met, the barrier will switch to
        the next phase, and the threads will proceed.

        Parameters
        ----------
        barrier: Expr | RegisterTensor
            The uint32 integer representing the address of the barrier in shared space. It can also be a register tensor
            with single element representing the address of the barrier.
        phase: Expr | RegisterTensor | int
            The phase value to wait for. It must be evaluated to either 0 or 1. It can also be a register tensor with single
            element representing the phase value.
        """
        self._builder.wait_barrier(barrier, phase)
