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

from hidet.ir.expr import Expr, as_expr

from tilus.ir.tensor import RegisterTensor

from .root import InstructionGroup


class BarrierInstructionGroup(InstructionGroup):
    producer_initial_phase = 1
    consumer_initial_phase = 0

    @typing.overload
    def alloc(self, count: Optional[Expr | int] = None) -> Expr: ...

    @typing.overload
    def alloc(self, count: Sequence[Expr | int]) -> RegisterTensor: ...

    def alloc(self, count: Sequence[Expr | int] | Optional[Expr | int] = None) -> RegisterTensor | Expr:
        """Allocate barrier(s) in shared memory and get its address in shared space.

         A barrier is an 64-bit data structure, encoded as uint64, in shared memory, which
         contains the following information in the 64 bits:

         - The current phase of the barrier (i.e., phase): 0 or 1.
         - The count of pending arrivals in the current phase: 1 to 2^20 - 1.
         - The count of expected arrivals in the next phase: 0 to 2^20 - 1.
         - The count of pending asynchronous memory transactions in the current phase (i.e., `tx-count`):
           -(2^20 - 1) to 2^20 - 1.

         This instruction allocates one or multiple uint64 elements in shared memory to be used as mbarrier(s).
         The parameter `count` specifies the expected arrivals for the barrier(s).
         After initialization, the barrier will have the following initial state:

         - phase = 0
         - pending arrivals = count[i]
         - expected arrivals = count[i]
         - tx-count = 0

         When `count` is not provided, it defaults to the number of threads in the current thread group.

         Some instructions (e.g., tma copy instructions) that take a barrier as an argument might:

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
        count: Sequence[Expr | int], Expr, int, optional
            The number of threads that must arrive at the barrier before any of them can proceed. It must be evaluated
            to positive int32 integer(s). When not provided, it defaults to the number of threads in the current thread group.
            When a sequence is provided, multiple barriers will be allocated with each barrier initialized with the corresponding
            expected arrivals specified in the sequence.

        Returns
        -------
        ret: RegisterTensor | Expr
            The shared memory address in shared space that points to the allocated barrier. The shared memory address has
            uint32 data type. When multiple barriers are allocated, a register tensor of shape (len(count),) and dtype uint32
            is returned, containing the shared memory addresses of the allocated barriers.
        """
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

    def arrive(self, barrier: Expr | RegisterTensor, count: Expr | int = 1) -> None:
        """Arrive at a barrier.

        Each thread in the current thread group decreases the pending arrivals of the given barrier by `count`.

        Parameters
        ----------
        barrier: Expr | RegisterTensor
            The uint32 integer representing the address of the barrier in shared space. It can also be a register tensor
            with single element representing the address of the barrier.
        count: Expr | int
            The number of arrivals contributed by each thread in the current thread group.  It must be evaluated to a positive int32.
            By default, it is 1.
        """
        self._builder.arrive_barrier(barrier, count=count)

    def arrive_and_expect_tx(self, barrier: Expr | RegisterTensor, tx_count: Expr | int) -> None:
        """Arrive at a barrier with expected asynchronous memory transactions.

        Each thread in the current thread group decreases the pending arrivals of the given barrier by 1, and increases the `expect-count` of the barrier by `tx_count`.

        Parameters
        ----------
        barrier: Expr | RegisterTensor
            The uint32 integer representing the address of the barrier in shared space. It can also be a register tensor with single element representing the address of the barrier.
        tx_count: Expr | int
            The number of asynchronous memory transactions expected to be issued by the threads in the current thread group after arriving at the barrier.
            It must be evaluated to a non-negative int32. The `expect-count` of the barrier will be increased by this number when the threads arrive at the barrier,
            and will be decreased by this number when the asynchronous memory transactions complete. Each thread will contribute to the same `tx_count` value,
            and the total `expect-count` increase will be `tx_count` * number of threads in the thread group.
        """
        self._builder.arrive_expect_tx_barrier(barrier, tx_count=tx_count)

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
