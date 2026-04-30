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
"""User-facing atomic tile operations exposed as ``self.atomic.*``.

See :doc:`/python-api/instruction-groups/atomic` for a detailed guide on the
element-wise vs. scatter forms and the ``sem`` / ``scope`` semantics.
"""

from __future__ import annotations

from typing import Optional

from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor

from .base import InstructionGroup


class AtomicInstructionGroup(InstructionGroup):
    """Tile-level atomic read-modify-write instructions.

    Two flavours are offered. The **element-wise** form (:meth:`shared_add` and
    friends) operates tile-to-tile: each lane contributes its own element of a
    register tile into the matching element of a shared or global tile, under
    the requested PTX ``atom.*`` operation. The **scatter** form
    (:meth:`shared_scatter_add` and friends) follows ``torch.scatter_add_``
    semantics --- a compile-time ``dim`` axis plus a per-lane ``indices`` tile
    picks the destination offset; the non-scatter axes come from the tile's
    global position.

    The full element-wise family is ``add`` / ``sub`` / ``min`` / ``max`` /
    ``exch`` / ``cas``. The scatter family drops ``exch`` and ``cas`` (their
    semantics under duplicate indices are unclear) and keeps just ``add`` /
    ``sub`` / ``min`` / ``max``.

    All methods accept two optional PTX qualifiers:

    - ``sem``: the memory-ordering qualifier. Candidates: ``'relaxed'``,
      ``'acquire'``, ``'release'``, ``'acq_rel'``. Defaults to ``'relaxed'``.
    - ``scope``: the sync-scope qualifier. Candidates: ``'cta'``, ``'cluster'``,
      ``'gpu'``, ``'sys'``. Defaults to ``'cta'`` on shared ops and ``'gpu'``
      on global ops.

    And an optional ``output`` register tensor that receives the per-element
    **pre-RMW** value at each target location. When no downstream code uses the
    returned register, the dead-code-elimination pass rewrites the instruction
    to carry ``output=None`` and codegen lowers to the destination-less
    ``red.*`` PTX form --- so the return value is free when unused.

    See Also
    --------
    :meth:`tilus.Script.store_shared_scatter` / :meth:`tilus.Script.store_global_scatter`
        Non-atomic scatter stores that share the same ``indices`` / ``values``
        contract. Use them when the scatter is guaranteed collision-free.
    """

    # ------------------------------------------------------------------
    # Element-wise atomics (shared)
    # ------------------------------------------------------------------

    def shared_add(
        self,
        dst: SharedTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = dst[i] + values[i]`` atomically, on shared memory.

        ``dst.shape`` and ``values.shape`` must be equal, and each lane contributes
        its own slice of ``values`` to the matching slice of ``dst`` with no
        broadcast or reduction.

        Parameters
        ----------
        dst: SharedTensor
            Destination tile in shared memory.
        values: RegisterTensor
            Per-lane contribution; same shape as ``dst``.
        sem: str
            PTX memory-ordering qualifier. Candidates: ``'relaxed'``, ``'acquire'``,
            ``'release'``, ``'acq_rel'``.
        scope: str
            PTX sync scope. Candidates: ``'cta'``, ``'cluster'``, ``'gpu'``,
            ``'sys'``.
        output: RegisterTensor, optional
            If provided, the per-element pre-RMW value at each location is written
            into this register tile (same shape as ``dst``).

        Returns
        -------
        RegisterTensor or None
            The pre-RMW register tile when ``output`` is consumed downstream; ``None``
            when unused (the DCE pass rewrites the instruction to the cheaper
            ``red.*`` form in that case).

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.add.s32`` (or ``red.*`` when the
          output is unused).
        """
        return self._builder.atomic_shared(
            dst=dst,
            values=values,
            op="add",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_sub(
        self,
        dst: SharedTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = dst[i] - values[i]`` atomically, on shared memory.

        PTX has no native ``atom.sub``; the codegen lowers this to ``atom.add``
        with the negated operand. See :meth:`shared_add` for the full parameter
        description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.add.s32`` with a negated input.
        """
        return self._builder.atomic_shared(
            dst=dst,
            values=values,
            op="sub",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_min(
        self,
        dst: SharedTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = min(dst[i], values[i])`` atomically, on shared memory.

        See :meth:`shared_add` for the full parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.min.s32``.
        """
        return self._builder.atomic_shared(
            dst=dst,
            values=values,
            op="min",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_max(
        self,
        dst: SharedTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = max(dst[i], values[i])`` atomically, on shared memory.

        See :meth:`shared_add` for the full parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.max.s32``.
        """
        return self._builder.atomic_shared(
            dst=dst,
            values=values,
            op="max",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_exch(
        self,
        dst: SharedTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``old = dst[i]; dst[i] = values[i]`` atomically (exchange).

        Unlike the arithmetic ops, ``exch`` has no ``red.*`` counterpart in PTX,
        so ``output`` is effectively always bound. Callers that don't need the
        old value should prefer a plain store.

        See :meth:`shared_add` for the full parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.exch.s32``.
        """
        return self._builder.atomic_shared(
            dst=dst,
            values=values,
            op="exch",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_cas(
        self,
        dst: SharedTensor,
        compare: RegisterTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise compare-and-swap on shared memory.

        Per element: ``old = dst[i]; if (old == compare[i]) dst[i] = values[i]``,
        atomically. The returned ``output`` (if bound) holds ``old``, which the
        caller typically inspects to decide whether the swap succeeded.

        Parameters
        ----------
        dst: SharedTensor
            Destination tile in shared memory.
        compare: RegisterTensor
            Expected-old-value tile; same shape and dtype as ``dst``.
        values: RegisterTensor
            Tile of replacement values; same shape and dtype as ``dst``.
        sem, scope, output
            See :meth:`shared_add`.

        Returns
        -------
        RegisterTensor or None
            Pre-CAS value at each element when ``output`` is consumed; ``None``
            otherwise. Note that, unlike the arithmetic ops, CAS has no ``red.*``
            form, so an unused output still costs a register allocation at the
            PTX level.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.cas.s32``.
        """
        return self._builder.atomic_shared(
            dst=dst,
            values=values,
            op="cas",
            sem=sem,
            scope=scope,
            output=output,
            compare=compare,
        )

    # ------------------------------------------------------------------
    # Element-wise atomics (global)
    # ------------------------------------------------------------------

    def global_add(
        self,
        dst: GlobalTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = dst[i] + values[i]`` atomically, on global memory.

        See :meth:`shared_add` for the full parameter description; the only
        difference is that ``dst`` is a :class:`~tilus.ir.GlobalTensor` and the
        default scope is ``'gpu'`` rather than ``'cta'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.add.s32`` (or ``red.*`` when the
          output is unused).
        """
        return self._builder.atomic_global(
            dst=dst,
            values=values,
            op="add",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_sub(
        self,
        dst: GlobalTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = dst[i] - values[i]`` atomically, on global memory.

        Lowered to ``atom.add`` with a negated operand; see :meth:`global_add`.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.add.s32`` with a negated input.
        """
        return self._builder.atomic_global(
            dst=dst,
            values=values,
            op="sub",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_min(
        self,
        dst: GlobalTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = min(dst[i], values[i])`` atomically, on global memory.

        See :meth:`global_add` for the full parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.min.s32``.
        """
        return self._builder.atomic_global(
            dst=dst,
            values=values,
            op="min",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_max(
        self,
        dst: GlobalTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise ``dst[i] = max(dst[i], values[i])`` atomically, on global memory.

        See :meth:`global_add` for the full parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.max.s32``.
        """
        return self._builder.atomic_global(
            dst=dst,
            values=values,
            op="max",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_exch(
        self,
        dst: GlobalTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise atomic exchange on global memory.

        ``old = dst[i]; dst[i] = values[i]``. See :meth:`shared_exch` for the
        caveat about ``exch`` having no ``red.*`` counterpart.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.exch.s32``.
        """
        return self._builder.atomic_global(
            dst=dst,
            values=values,
            op="exch",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_cas(
        self,
        dst: GlobalTensor,
        compare: RegisterTensor,
        values: RegisterTensor,
        *,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Element-wise compare-and-swap on global memory.

        Per element: ``old = dst[i]; if (old == compare[i]) dst[i] = values[i]``.
        See :meth:`shared_cas` for the full parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.cas.s32``.
        """
        return self._builder.atomic_global(
            dst=dst,
            values=values,
            op="cas",
            sem=sem,
            scope=scope,
            output=output,
            compare=compare,
        )

    # ------------------------------------------------------------------
    # Scatter atomics (shared)
    # ------------------------------------------------------------------

    def shared_scatter_add(
        self,
        dst: SharedTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-add into a shared tile along ``dim``.

        For each tile element *k*, performs
        ``dst[..., indices[k], ...] = dst[..., indices[k], ...] + values[k]``
        atomically, where ``indices`` picks positions along ``dim`` and the
        non-scatter axes come from the lane's own tile position.

        ``indices.shape == values.shape`` strictly (identical RegisterLayout);
        ``dst``'s non-``dim`` axes must match ``indices`` exactly. Out-of-range
        index values are undefined --- there is no runtime bounds check.

        Parameters
        ----------
        dst: SharedTensor
            Destination tile in shared memory.
        dim: int
            Compile-time scatter axis into ``dst``.
        indices: RegisterTensor
            Per-lane integer indices along ``dim``.
        values: RegisterTensor
            Per-lane contributions; same shape and layout as ``indices``.
        sem: str
            PTX memory-ordering qualifier. See :class:`AtomicInstructionGroup`
            for the accepted values.
        scope: str
            PTX sync scope. See :class:`AtomicInstructionGroup`.
        output: RegisterTensor, optional
            If provided, receives the per-element pre-RMW value at each
            scattered location (same shape as ``indices``).

        Returns
        -------
        RegisterTensor or None
            Pre-RMW values when ``output`` is consumed downstream; ``None``
            when unused (the DCE pass rewrites the instruction to the cheaper
            ``red.*`` form).

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.add.s32`` (or ``red.*`` when the
          output is unused).
        """
        return self._builder.atomic_shared_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="add",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_scatter_sub(
        self,
        dst: SharedTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-sub into a shared tile along ``dim``; lowered to ``atom.add`` with a negated value.

        See :meth:`shared_scatter_add` for the parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.add.s32`` with a negated input.
        """
        return self._builder.atomic_shared_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="sub",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_scatter_min(
        self,
        dst: SharedTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-min into a shared tile along ``dim``.

        See :meth:`shared_scatter_add` for the parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.min.s32``.
        """
        return self._builder.atomic_shared_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="min",
            sem=sem,
            scope=scope,
            output=output,
        )

    def shared_scatter_max(
        self,
        dst: SharedTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-max into a shared tile along ``dim``.

        See :meth:`shared_scatter_add` for the parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.shared.max.s32``.
        """
        return self._builder.atomic_shared_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="max",
            sem=sem,
            scope=scope,
            output=output,
        )

    # ------------------------------------------------------------------
    # Scatter atomics (global)
    # ------------------------------------------------------------------

    def global_scatter_add(
        self,
        dst: GlobalTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-add into a global tile along ``dim``.

        Same contract as :meth:`shared_scatter_add` but the destination is a
        :class:`~tilus.ir.GlobalTensor` and the default scope is ``'gpu'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.add.s32`` (or ``red.*`` when the
          output is unused).
        """
        return self._builder.atomic_global_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="add",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_scatter_sub(
        self,
        dst: GlobalTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-sub into a global tile along ``dim``; lowered to ``atom.add`` with a negated value.

        See :meth:`shared_scatter_add` for the parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.add.s32`` with a negated input.
        """
        return self._builder.atomic_global_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="sub",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_scatter_min(
        self,
        dst: GlobalTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-min into a global tile along ``dim``.

        See :meth:`shared_scatter_add` for the parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.min.s32``.
        """
        return self._builder.atomic_global_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="min",
            sem=sem,
            scope=scope,
            output=output,
        )

    def global_scatter_max(
        self,
        dst: GlobalTensor,
        *,
        dim: int,
        indices: RegisterTensor,
        values: RegisterTensor,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> Optional[RegisterTensor]:
        """Scatter-max into a global tile along ``dim``.

        See :meth:`shared_scatter_add` for the parameter description.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70).
        - **PTX**: ``atom.{sem}.{scope}.global.max.s32``.
        """
        return self._builder.atomic_global_scatter(
            dst=dst,
            indices=indices,
            values=values,
            dim=dim,
            op="max",
            sem=sem,
            scope=scope,
            output=output,
        )
