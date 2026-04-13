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
from typing import Optional, Sequence

from tilus.hidet.ir.expr import Expr
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor

from .root import InstructionGroup


class TmaInstructionGroup(InstructionGroup):
    """Tensor Memory Accelerator (TMA) instructions for asynchronous bulk data transfers.

    TMA is a dedicated hardware engine on Hopper+ GPUs that asynchronously copies multi-dimensional
    tiles between global memory and shared memory, without occupying SM compute resources.

    TMA transfers are **asynchronous**: the issuing thread returns immediately while the TMA engine
    performs the copy in the background. Completion is tracked through mbarriers:

    1. Issue ``global_to_shared()`` or ``shared_to_global()`` — the mbarrier's tx-count is
       automatically increased.
    2. The TMA engine completes the transfer and decrements the mbarrier's tx-count.
    3. Consumers call ``mbarrier.wait()`` to block until all transfers for a phase are done.

    For legacy (non-mbarrier) async copies, use ``commit_group()`` and ``wait_group()`` to
    group and synchronize transfers.

    TMA supports **multicast** (``multicast_mask``) to deliver the same global tile to shared
    memory of multiple CTAs in a cluster, and **CTA groups** (``cta_group=2``) for coordinated
    two-CTA operations where the mbarrier can reside on a peer CTA.
    """

    def global_to_shared(
        self,
        *,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        mbarrier: Expr | RegisterTensor,
        cta_group: int = 1,
        multicast_mask: Optional[Expr | int] = None,
        cache_policy: Optional[Expr] = None,
    ) -> None:
        """Asynchronously copy a tile from global memory to shared memory via TMA.

        Issues an asynchronous TMA transfer from a region of ``src`` (global) to ``dst`` (shared).
        The ``offsets`` specify where in the global tensor the tile starts, and ``dims`` specifies
        which global dimensions map to the shared tensor dimensions.

        Completion is tracked via the ``mbarrier``: this instruction automatically increases the
        barrier's tx-count by the transfer size in bytes. When the TMA engine finishes, it
        decrements the tx-count by the same amount. Use ``mbarrier.wait()`` to block until done.

        **Multicast and CTA groups:**

        - ``cta_group=1, multicast_mask=None``: single-CTA transfer. Both ``dst`` and ``mbarrier``
          must be in the current CTA.
        - ``cta_group=1, multicast_mask != None``: the loaded tile is delivered to shared memory of
          all CTAs specified by the mask. ``mbarrier`` must be in the current CTA.
        - ``cta_group=2, multicast_mask=None``: ``dst`` must be in the current CTA, but ``mbarrier``
          can be in the current or peer CTA.
        - ``cta_group=2, multicast_mask != None``: the tile is multicast, and ``mbarrier`` can be in
          the current or peer CTA. Barriers at the same shared memory offset in the target CTAs
          are signaled.

        Parameters
        ----------
        src: GlobalTensor
            The global tensor to copy from.
        dst: SharedTensor
            The shared tensor to copy to.
        offsets: Sequence[Expr | int]
            Starting offsets for each dimension of the global tensor. Length must match the rank
            of the global tensor.
        dims: Sequence[int], optional
            Which dimensions of the global tensor are being sliced. ``dims[0]`` maps to the first
            dimension of the shared tensor, ``dims[1]`` to the second, etc. If not provided,
            defaults to all dimensions in order.
        mbarrier: Expr | RegisterTensor
            The barrier for tracking completion. A uint32 expression or single-element register
            tensor containing the barrier's shared memory address.
        cta_group: int
            CTA group size for the transfer. 1 (default) for single-CTA, 2 for two-CTA
            coordinated operations.
        multicast_mask: Optional[Expr | int]
            A uint16 bitmask specifying which CTAs in the cluster receive the data. Bit *i*
            corresponds to the CTA with rank *i*. When ``None``, no multicast is performed.
        cache_policy: Optional[Expr]
            Cache eviction policy encoded as a uint64 value.

        Notes
        -----
        - **Thread group**: Must be executed by a warp-aligned thread group (i.e., a multiple of 32 threads).
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``cp.async.bulk.tensor.global.shared::cta.tile.mbarrier::complete_tx::bytes``
        """
        self._builder.copy_async_tensor_global_to_shared(
            src=src,
            dst=dst,
            offsets=offsets,
            dims=dims,
            mbarrier=mbarrier,
            cta_group=cta_group,
            multicast_mask=multicast_mask,
            cache_policy=cache_policy,
        )

    def shared_to_global(
        self,
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        cache_policy: Optional[Expr] = None,
    ) -> None:
        """Asynchronously copy a tile from shared memory to global memory via TMA.

        Issues an asynchronous TMA transfer from ``src`` (shared) to a region of ``dst`` (global).
        The ``offsets`` specify where in the global tensor the tile is written to, and ``dims``
        specifies which global dimensions map to the shared tensor dimensions.

        Unlike ``global_to_shared()``, this instruction does not use an mbarrier. Use
        ``commit_group()`` and ``wait_group()`` to synchronize completion.

        .. important::

            If the shared memory data was written via the **generic proxy** (e.g., ``store_shared()``),
            a ``fence.proxy_async()`` or ``fence.proxy_async_release()`` must be called before this
            instruction to ensure the writes are visible to the TMA engine (async proxy).

        Parameters
        ----------
        src: SharedTensor
            The shared tensor to copy from.
        dst: GlobalTensor
            The global tensor to copy to.
        offsets: Sequence[Expr | int]
            Starting offsets for each dimension of the global tensor. Length must match the rank
            of the global tensor.
        dims: Sequence[int], optional
            Which dimensions of the global tensor are being sliced. If not provided, defaults to
            all dimensions in order.
        cache_policy: Optional[Expr]
            Cache eviction policy encoded as a uint64 value.

        Notes
        -----
        - **Thread group**: Must be executed by a warp-aligned thread group (i.e., a multiple of 32 threads).
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``cp.async.bulk.tensor.shared::cta.global.tile``
        """
        self._builder.copy_async_tensor_shared_to_global(
            src=src, dst=dst, offsets=offsets, dims=dims, cache_policy=cache_policy
        )

    def commit_group(self):
        """Commit pending TMA async copy operations into a group.

        Groups all prior uncommitted ``shared_to_global()`` operations so they can be
        collectively waited on with ``wait_group()``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``cp.async.bulk.commit_group``
        """
        self._builder.copy_async_tensor_commit_group()

    def wait_group(self, n: int, read: bool = False) -> None:
        """Wait for TMA async copy commit groups to complete.

        Blocks until at most ``n`` commit groups remain pending. Use ``n=0`` to wait for all
        committed groups.

        When ``read=False`` (default), waits for all operations to complete, including writes
        being visible to the executing thread.

        When ``read=True``, only waits for reads from source locations to complete. This is
        useful when the source shared memory needs to be reused, but there is no subsequent
        instruction that reads the destination global memory. If subsequent instructions need
        to read the global memory written by TMA, use the default ``read=False``.

        Parameters
        ----------
        n: int
            The number of groups to allow to be on-the-fly. It should be an integer larger or equal to 0.
        read: bool
            If True, only wait for reads to complete (not writes). Default is False.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``cp.async.bulk.wait_group`` or ``cp.async.bulk.wait_group.read``
        """
        self._builder.copy_async_tensor_wait_group(n, read=read)
