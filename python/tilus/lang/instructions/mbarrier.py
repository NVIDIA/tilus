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

    An mbarrier is a 64-bit synchronization primitive in shared memory that tracks two counts:
    **pending arrivals** and **pending transaction bytes (tx-count)**. A barrier transitions to the
    next phase when both counts reach zero.

    Mbarriers coordinate producer-consumer patterns in pipelined GPU kernels. The typical workflow is:

    1. **Allocate** barriers with ``alloc()``, specifying expected arrival counts per stage.
    2. **Producers** call ``arrive()`` or ``arrive_and_expect_tx()`` to signal progress and declare
       expected async data transfers (e.g., TMA copies). The TMA engine automatically decrements
       the tx-count as transfers complete.
    3. **Consumers** call ``wait()`` to block until the barrier's current phase finishes (both
       pending arrivals and tx-count reach zero).

    The barrier alternates between phase 0 and phase 1. Use ``producer_initial_phase`` and
    ``consumer_initial_phase`` as starting phase values for multi-stage pipelines.

    For cluster-wide synchronization, ``arrive_and_expect_tx_multicast()`` signals barriers across
    multiple CTAs, and ``arrive_and_expect_tx_remote()`` signals a specific peer CTA's barrier.
    """

    producer_initial_phase = 1
    consumer_initial_phase = 0

    def alloc(self, counts: Sequence[Expr | int] | Expr | int) -> RegisterTensor:
        """Allocate and initialize one or more mbarriers in shared memory.

        Each barrier is initialized with phase 0, the specified expected arrival count, and
        tx-count of 0. When a single value is provided, one barrier is allocated; when a
        sequence is provided, multiple barriers are allocated with the corresponding counts.

        Parameters
        ----------
        counts: Sequence[Expr | int] | Expr | int, optional
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

        Signals an arrive-and-expect-tx on the barriers at the same shared memory offset in each
        CTA specified by ``multicast_mask``. The arrival count is 1 and the tx-count is increased
        by ``transaction_bytes`` on each signaled barrier. One thread is elected per CTA to perform
        the operation.

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

        Signals an arrive-and-expect-tx on the barrier in the CTA specified by ``target_rank``.
        The barrier at the same shared memory offset as the local ``barrier`` in the target CTA
        is signaled with arrival count 1 and tx-count increased by ``transaction_bytes``.

        Parameters
        ----------
        barrier: RegisterTensor
            A single-element uint32 register tensor with the barrier's shared memory address
            in the current CTA. The same offset is used for the target CTA.
        transaction_bytes: Expr | int
            Expected async transfer size in bytes. Must evaluate to a non-negative int32.
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
