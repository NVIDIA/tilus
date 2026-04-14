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

from tilus.hidet.ir.expr import Expr, as_expr
from tilus.ir.tensor import RegisterTensor

from .root import InstructionGroup


class BarrierInstructionGroup(InstructionGroup):
    """Memory barrier (mbarrier) instructions for synchronizing asynchronous operations.

    See :doc:`/python-api/instruction-groups/mbarrier` for a detailed guide on how mbarriers work.
    """

    #: Initial phase value for the producer in a producer-consumer pipeline.
    #: The producer starts by waiting for the barrier to leave this phase (i.e., waiting for the
    #: consumer to free the slot), so it is initialized to ``1``.
    producer_initial_phase = 1

    #: Initial phase value for the consumer in a producer-consumer pipeline.
    #: The consumer starts by waiting for the barrier to leave this phase (i.e., waiting for the
    #: producer to fill the slot), so it is initialized to ``0`` (the barrier's initial phase).
    consumer_initial_phase = 0

    def alloc(self, counts: Sequence[Expr | int] | Expr | int) -> RegisterTensor:
        """Allocate and initialize one or more mbarriers in shared memory.

        Each barrier is a 64-bit object in shared memory, initialized with:

        - ``phase = 0``
        - ``pending_arrivals = counts[i]`` (the expected arrival count)
        - ``expected_arrivals = counts[i]`` (used to reset pending_arrivals on phase flip)
        - ``tx-count = 0``

        A single value allocates one barrier; a sequence allocates multiple barriers.

        Parameters
        ----------
        counts: Sequence[Expr | int] | Expr | int
            Expected arrival counts for the barriers. Each count must evaluate to a positive
            int32. A single value allocates one barrier; a sequence allocates multiple.

        Returns
        -------
        ret: RegisterTensor
            A register tensor of dtype uint32 containing the shared memory address(es) of
            the allocated barrier(s). Element *i* holds the address for ``counts[i]``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``mbarrier.init.shared::cta.b64``
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

        Each thread in the current thread group decrements the barrier's pending arrival count
        by ``count``. When pending arrivals (and tx-count) both reach zero, the hardware flips
        the phase and resets the counters for the next phase.

        With ``sem='release'``, all prior memory writes by the arriving thread are guaranteed
        visible to any thread that later completes a successful ``wait`` with acquire semantics
        on this barrier.

        Parameters
        ----------
        barrier: RegisterTensor
            A single-element uint32 register tensor holding the barrier's shared memory address.
        count: Expr | int
            The number of arrivals contributed by each thread. Must evaluate to a positive int32.
            Default is 1.
        sem: str
            Memory ordering semantics. ``'release'`` ensures prior writes are visible to waiters;
            ``'relaxed'`` provides no ordering guarantees. Candidates: ``'relaxed'``, ``'release'``.
        scope: str
            Synchronization scope. Candidates: ``'cta'``, ``'cluster'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``mbarrier.arrive.shared::cta.b64``
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
        """Arrive at a barrier and declare expected asynchronous transaction bytes.

        Each thread in the current thread group performs two updates on the barrier:

        1. Decrements the pending arrival count by 1.
        2. Increases the pending tx-count by ``transaction_bytes``.

        The tx-count tracks asynchronous data transfers (e.g., TMA copies). When an async
        operation tied to this barrier completes, the hardware automatically decrements the
        tx-count by the number of bytes transferred. The phase completes only when both
        pending arrivals and tx-count reach zero.

        Typically used with :meth:`~tilus.Script.single_thread` so that only one thread
        sets the tx-count expectation, while the TMA engine performs the actual transfer.

        Parameters
        ----------
        barrier: RegisterTensor
            A single-element uint32 register tensor holding the barrier's shared memory address.
        transaction_bytes: Expr | int
            The number of bytes expected from async transactions (e.g., TMA copies). The
            barrier's tx-count is increased by this value; it is automatically decreased by
            the hardware as the transactions complete. Must evaluate to a non-negative int32.
        sem: str
            Memory ordering semantics. Candidates: ``'relaxed'``, ``'release'``.
        scope: str
            Synchronization scope. Candidates: ``'cta'``, ``'cluster'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``mbarrier.arrive.expect_tx.shared::cta.b64``
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
        """Wait for a barrier phase to complete.

        All threads in the current thread group block until the barrier's current phase
        differs from ``phase``. This means the phase has completed --- all pending arrivals
        and tx-count have reached zero, the hardware has flipped the phase bit, and it is
        safe to read data produced in that phase.

        If the barrier's current phase already differs from ``phase`` (i.e., the phase has
        already completed), the threads proceed immediately without blocking.

        With ``sem='acquire'``, all writes made visible by arrive operations (with release
        semantics) on this barrier in the completed phase are guaranteed visible to the
        waiting threads.

        Parameters
        ----------
        barrier: RegisterTensor
            A single-element uint32 register tensor holding the barrier's shared memory address.
        phase: Expr | RegisterTensor | int
            The phase to wait for completion of. Must be 0 or 1. Can also be a single-element
            register tensor.
        sem: str
            Memory ordering semantics. ``'acquire'`` ensures writes from the completed phase
            are visible; ``'relaxed'`` provides no ordering. Candidates: ``'acquire'``, ``'relaxed'``.
        scope: str
            Synchronization scope. Candidates: ``'cta'``, ``'cluster'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``mbarrier.try_wait.parity.shared::cta.b64``
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
        """Arrive at barriers across multiple CTAs with expected async transactions.

        Unlike :meth:`arrive` and :meth:`arrive_and_expect_tx` where every thread in the group
        arrives on the same barrier, this instruction elects **one thread per target CTA** in
        ``multicast_mask``. Each elected thread arrives on the barrier at the same shared memory
        offset in its assigned CTA. The arrival count is 1 and the tx-count is increased by
        ``transaction_bytes`` on each signaled barrier.

        Parameters
        ----------
        barrier: RegisterTensor
            A single-element uint32 register tensor with the barrier's shared memory address
            in the current CTA. The same offset is used for peer CTAs.
        transaction_bytes: Expr | int
            Expected async transfer size in bytes. Must evaluate to a non-negative int32.
        multicast_mask: int
            Bitmask of CTAs to signal. Bit *i* corresponds to the CTA with rank *i*.
            E.g., ``0b101`` signals CTAs 0 and 2.
        sem: str
            Memory ordering semantics. Candidates: ``'relaxed'``, ``'release'``.
        scope: str
            Synchronization scope. Candidates: ``'cta'``, ``'cluster'``.

        Notes
        -----
        - **Thread group**: Must be executed by a thread group with at least 16 threads.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``mbarrier.arrive.expect_tx.shared::cluster.b64`` with ``mapa.shared::cluster``
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
        """Arrive at a peer CTA's barrier with expected async transactions.

        Each thread in the current thread group arrives on the barrier in the remote CTA
        specified by ``target_rank``, using the same shared memory offset as the local
        ``barrier``. Each thread decrements the remote barrier's pending arrival count by 1
        and increases its tx-count by ``transaction_bytes``.

        This is used in cluster-wide pipelines where one CTA needs to signal another CTA's
        barrier (e.g., to indicate that data has been loaded into the remote CTA's shared
        memory).

        Parameters
        ----------
        barrier: RegisterTensor
            A single-element uint32 register tensor with the barrier's shared memory address
            in the current CTA. The barrier at the same offset in the target CTA is signaled.
        transaction_bytes: Expr | int
            Expected async transfer size in bytes. The remote barrier's tx-count is increased
            by this value. Must evaluate to a non-negative int32.
        target_rank: int
            Rank of the target CTA in the cluster. Must be in ``[0, clusterSize)``.
        sem: str
            Memory ordering semantics. Candidates: ``'relaxed'``, ``'release'``.
        scope: str
            Synchronization scope. Candidates: ``'cta'``, ``'cluster'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``mbarrier.arrive.expect_tx.shared::cluster.b64`` with ``mapa.shared::cluster``
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
