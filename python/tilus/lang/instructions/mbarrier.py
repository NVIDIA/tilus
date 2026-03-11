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
from typing import Sequence

from hidet.ir.expr import Expr, as_expr

from tilus.ir.tensor import RegisterTensor

from .root import InstructionGroup


class BarrierInstructionGroup(InstructionGroup):
    producer_initial_phase = 1
    consumer_initial_phase = 0

    def alloc(self, counts: Sequence[Expr | int] | Expr | int) -> RegisterTensor:
        """Allocate barriers in shared memory and get its address in shared space.

         A barrier is an 64-bit data structure, encoded as uint64, in shared memory, which
         contains the following information in the 64 bits:

         - The current phase of the barrier (i.e., phase): 0 or 1.
         - The count of pending arrivals in the current phase: 1 to 2^20 - 1.
         - The count of expected arrivals in the next phase: 0 to 2^20 - 1.
         - The count of pending asynchronous memory transactions in the current phase (i.e., `tx-count`):
           -(2^20 - 1) to 2^20 - 1.

         This instruction allocates one or multiple uint64 elements in shared memory to be used as mbarrier(s).
         The parameter `counts` specifies the expected arrivals for the barrier(s).
         After initialization, the barrier will have the following initial state:

         - phase = 0
         - pending arrivals = counts[i]
         - expected arrivals = counts[i]
         - tx-count = 0

         When `counts` is not provided, it defaults to [NUM_THREADS] where NUM_THREADS is the number of threads in the current thread group.

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
        counts: Sequence[Expr | int] | Expr | int, optional
            The sequence of expected arrival counts for the allocated barriers. Each count must be evaluated to a positive int32 integer.
            When not provided, it defaults to [NUM_THREADS] where NUM_THREADS is the number of threads in the current thread group.
            Multiple barriers will be allocated with each barrier initialized with the corresponding expected arrivals specified in the sequence.
            When a single Expr or int is provided, it will be treated as a sequence with one element, and one barrier will be allocated and initialized
            with the given expected arrival count.

        Returns
        -------
        ret: RegisterTensor
            A register tensor of dtype uint32 containing the address of the allocated barrier(s) in shared memory.
            The i-th element corresponds to the address of the barrier with expected arrivals specified by counts[i].
        """
        if isinstance(counts, (Expr, int)):
            processed_counts = [as_expr(counts)]
        else:
            processed_counts = [as_expr(c) if isinstance(c, (Expr, int)) else None for c in counts]
        return self._builder.allocate_barrier(processed_counts)

    def arrive(
        self,
        barrier: RegisterTensor,
        count: Expr | int = 1,
        sem: str = "release",
        scope: str = "cta",
    ) -> None:
        """Arrive at a barrier.

        Each thread in the current thread group decreases the pending arrivals of the given barrier by `count`.

        Parameters
        ----------
        barrier: RegisterTensor
            A register tensor with one uint32 element representing the address of the barrier in the local shared memory.
        count: Expr | int
            The number of arrivals contributed by each thread in the current thread group. It must be evaluated to a positive int32.
            By default, it is 1.
        sem: str
            The memory ordering semantics for the arrive operation. Candidates: 'relaxed', 'release'.
        scope: str
            The syncrhonization scope for the arrive operation. Candidates: 'cta', 'cluster'.
        """
        if sem not in ("relaxed", "release"):
            raise ValueError(
                f"Invalid memory ordering semantics for arrive operation: {sem}. Supported candidates are 'relaxed' and 'release'."
            )
        if scope not in ("cta", "cluster"):
            raise ValueError(
                f"Invalid scope for arrive operation: {scope}. Supported candidates are 'cta' and 'cluster'."
            )
        self._builder.arrive_barrier(barrier, count=count, sem=sem, scope=scope)

    def arrive_and_expect_tx(
        self,
        barrier: RegisterTensor,
        transaction_bytes: Expr | int,
        sem: str = "release",
        scope: str = "cta",
    ) -> None:
        """Arrive at a barrier with expected asynchronous memory transactions.

        Each thread in the current thread group decreases the pending arrivals of the given barrier by 1, and increases
        the barrier's pending transaction byte count (tx-count) by `transaction_bytes`.

        Parameters
        ----------
        barrier: RegisterTensor
            A register tensor with one uint32 element representing the address of the barrier in the local shared memory.
        transaction_bytes: Expr | int
            The number of bytes expected to be delivered by asynchronous memory transactions (e.g., TMA copies) to this
            barrier. The barrier's tx-count will be increased by this value on arrival and decreased as the async
            transactions complete. It must be evaluated to a non-negative int32.
        sem: str
            The memory ordering semantics for the arrive operation. Candidates: 'relaxed', 'release'.
        scope: str
            The syncrhonization scope for the arrive operation. Candidates: 'cta', 'cluster'.
        """
        if sem not in ("relaxed", "release"):
            raise ValueError(
                f"Invalid memory ordering semantics for arrive operation: {sem}. Supported candidates are 'relaxed' and 'release'."
            )
        if scope not in ("cta", "cluster"):
            raise ValueError(
                f"Invalid scope for arrive operation: {scope}. Supported candidates are 'cta' and 'cluster'."
            )
        self._builder.arrive_expect_tx_barrier(barrier, transaction_bytes=transaction_bytes, sem=sem, scope=scope)

    def wait(
        self,
        barrier: RegisterTensor,
        phase: Expr | RegisterTensor | int,
        sem: str = "acquire",
        scope: str = "cta",
    ) -> None:
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
        barrier: RegisterTensor
            The barrier to wait on. It must be a register tensor with one uint32 element representing the address of the barrier in the local shared memory.
        phase: Expr | RegisterTensor | int
            The phase value to wait for. It must be evaluated to either 0 or 1. It can also be a register tensor with single
            element representing the phase value.
        sem: str
            The memory ordering semantics for the wait operation. Candidates: 'acquire', 'relaxed'.
        scope: str
            The synchronization scope for the wait operation. Candidates: 'cta', 'cluster'.
        """
        if sem not in ("acquire", "relaxed"):
            raise ValueError(
                f"Invalid memory ordering semantics for wait operation: {sem}. Supported candidates are 'acquire' and 'relaxed'."
            )
        if scope not in ("cta", "cluster"):
            raise ValueError(
                f"Invalid scope for wait operation: {scope}. Supported candidates are 'cta' and 'cluster'."
            )
        self._builder.wait_barrier(barrier, phase, sem=sem, scope=scope)

    def arrive_and_expect_tx_multicast(
        self,
        barrier: RegisterTensor,
        transaction_bytes: Expr | int,
        multicast_mask: int,
        sem: str = "release",
        scope: str = "cluster",
    ) -> None:
        """
        Arrive at a barrier with expected asynchronous memory transactions and multicast the arrival to a subset of blocks in the cluster.

        This instruction arrives the all barriers on each CTA in the current block cluster with the same offset of the given barrier. The arrival count is 1
        and the expected transaction bytes are specified by `transaction_bytes`.

        The `barrier` parameter specifies the address of the barrier for the current block, and the `multicast_mask` parameter specifies which blocks
        in the cluster will be involved in this arrival operation. For example, when `multicast_mask` is 0b101, only the barriers on blocks 0 and 2 be signaled by the arrival
        and expect-tx operation, while other blocks will not be affected.

        This instruction must be executed in a thread group with at least 16 threads. The arrive-and-expect-tx operation will not be performed by each
        thread. Instead, one thread will be elected to perform the operation for one CTA in the multicast.

        Parameters
        ----------
        barrier: RegisterTensor
            A register tensor with one uint32 element representing the address of the barrier in the local shared memory for the current block.
        transaction_bytes: Expr | int
            The number of bytes expected to be delivered by asynchronous memory transactions (e.g., TMA copies) to the barriers. The barriers' tx-count
            will be increased by this value on arrival and decreased as the async transactions complete. It must be evaluated to a non-negative int32.
        multicast_mask: int
            A bitmask specifying which blocks in the cluster will be involved in this arrival operation. The least significant bit corresponds to block 0,
            the second least significant bit corresponds to block 1, and so on. For example, when `multicast_mask` is 0b101, only the barriers on
            blocks 0 and 2 be signaled by the arrival and expect-tx operation, while other blocks will not be affected.
        sem: str
            The memory ordering semantics for the arrive operation. Candidates: 'relaxed', 'release'.
        scope: str
            The syncrhonization scope for the arrive operation. Candidates: 'cta', 'cluster'.
        """
        if sem not in ("relaxed", "release"):
            raise ValueError(
                f"Invalid memory ordering semantics for arrive operation: {sem}. Supported candidates are 'relaxed' and 'release'."
            )
        if scope not in ("cta", "cluster"):
            raise ValueError(
                f"Invalid scope for arrive operation: {scope}. Supported candidates are 'cta' and 'cluster'."
            )
        self._builder.arrive_expect_tx_multicast_barrier(
            barrier, transaction_bytes=transaction_bytes, multicast_mask=multicast_mask, sem=sem, scope=scope
        )

    def arrive_and_expect_tx_remote(
        self,
        barrier: RegisterTensor,
        transaction_bytes: Expr | int,
        target_rank: int,
        sem: str = "release",
        scope: str = "cluster",
    ) -> None:
        """
        Arrive at a barrier on a peer thread block in the same cluster with expected asynchronous memory transactions.

        The arrival count would be 1 and the expected transaction bytes are specified by `transaction_bytes`.

        The `barrier` parameter must be on the local shared memory of the current block. The mbarrier on the target block
        specified by `target_rank` that has the same offset as the given `barrier` will be signaled by this arrival operation
        and expect the asynchronous transactions.

        Parameters
        ----------
        barrier: RegisterTensor
            A register tensor with one uint32 element representing the address of the barrier in the local shared memory for the current block.
        transaction_bytes: Expr | int
            The number of bytes expected to be delivered by asynchronous memory transactions (e.g., TMA copies) to the barrier. The barrier's tx-count
            will be increased by this value on arrival and decreased as the async transactions complete. It must be evaluated to a non-negative int32.
        target_rank: int
            The rank of the target block in the same cluster for this remote arrive operation. It must be in the range of `[0, clusterSize)` where
            `clusterSize` is the total number of blocks in the cluster.
        sem: str
            The memory ordering semantics for the arrive operation. Candidates: 'relaxed', 'release'.
        scope: str
            The syncrhonization scope for the arrive operation. Candidates: 'cta', 'cluster'.
        """
        if sem not in ("relaxed", "release"):
            raise ValueError(
                f"Invalid memory ordering semantics for arrive operation: {sem}. Supported candidates are 'relaxed' and 'release'."
            )
        if scope not in ("cta", "cluster"):
            raise ValueError(
                f"Invalid scope for arrive operation: {scope}. Supported candidates are 'cta' and 'cluster'."
            )
        self._builder.arrive_expect_tx_remote_barrier(
            barrier, transaction_bytes=transaction_bytes, target_rank=target_rank, sem=sem, scope=scope
        )
