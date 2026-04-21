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
"""Layout-agnostic prefix scan via Blelloch's algorithm.

Blelloch is bit-local: every round's data movement is between positions ``i``
and ``i XOR 2**d`` for a single bit ``d``. That single bit maps cleanly to one
level of the register layout --- a local-idx bit (intra-thread register
combine), a lane-id bit (``shfl_xor_sync``), or a warp-id bit (shared-memory
exchange). So the same algorithm works for any tilus register layout,
including interleaved dim-axis bit orderings, without layout-classification
heuristics.

Cost: ``2 * log2(dim_extent)`` rounds (one up-sweep + one down-sweep), each
round a single primitive.

Kogge-Stone-style specialization for grouped layouts is a reasonable
follow-up optimization but not needed for correctness or the first pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet import boolean
from tilus.hidet.ir import DataType
from tilus.hidet.ir.dtypes import int32, uint32
from tilus.hidet.ir.expr import Expr, Var, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, cast, logical_and
from tilus.hidet.ir.primitives.cuda.shfl import shfl_xor_sync
from tilus.hidet.ir.type import tensor_pointer_type
from tilus.hidet.ir.utils.index_transform import index_deserialize, index_serialize
from tilus.ir.instructions.generic import ScanInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.tensor import RegisterTensor
from tilus.target import nvgpu_any

# ---------------------------------------------------------------------------
# Op semantics
# ---------------------------------------------------------------------------


def _identity(op: str, dtype: DataType) -> Expr:
    if op == "add":
        return dtype.zero
    if op == "mul":
        return dtype.one
    if op == "max":
        return dtype.min_value
    if op == "min":
        return dtype.max_value
    if op == "or":
        return dtype.zero
    if op == "xor":
        return dtype.zero
    if op == "and":
        # All-ones of the dtype — identity for bitwise AND.
        return bitwise_not(dtype.zero)
    raise NotImplementedError(f"scan op {op!r} has no registered identity")


def _combine(op: str, lhs: Expr, rhs: Expr) -> Expr:
    from tilus.hidet.ir.primitives import max as prim_max
    from tilus.hidet.ir.primitives import min as prim_min

    if op == "add":
        return lhs + rhs
    if op == "mul":
        return lhs * rhs
    if op == "max":
        return prim_max(lhs, rhs)
    if op == "min":
        return prim_min(lhs, rhs)
    if op == "and":
        return bitwise_and(lhs, rhs)
    if op == "or":
        return bitwise_or(lhs, rhs)
    if op == "xor":
        return bitwise_xor(lhs, rhs)
    raise NotImplementedError(f"scan op {op!r} has no registered combine")


# ---------------------------------------------------------------------------
# Bit classification
# ---------------------------------------------------------------------------

DimBitLevel = Literal["local", "lane", "warp"]


@dataclass(frozen=True)
class DimBit:
    """One dim-axis bit, classified by where it lives in the layout.

    ``level`` says which coord space this bit belongs to, and ``pos`` is the
    bit's position within that coord space.

    For ``level == "local"`` or ``level == "warp"``, ``partner_delta_in_local``
    is the XOR delta in the local-idx serialization (for local bits) or left
    unused (warp bits use ``pos`` directly). For lane bits ``pos`` is the lane
    bit to XOR via shfl.
    """

    level: DimBitLevel
    pos: int  # bit position within local_idx / lane_id / warp_id


def _classify_dim_bits(layout: RegisterLayout, dim: int) -> list[DimBit]:
    """Return the dim-axis bits of ``dim`` in dim-axis order (low to high).

    Walks ``grouped_modes[dim]`` from innermost mode (smallest stride in the
    shape axis) outward; within each mode, from LSB to MSB. Each bit is
    classified as living in a local_idx, lane_id, or warp_id bit position.
    """
    dim_modes = layout.grouped_modes[dim]

    # For each mode, precompute its bit-range in local_idx or spatial_id.
    local_bit_start: dict[int, int] = {}
    offset = 0
    for lm in reversed(layout.local_modes):
        num_bits = int(math.log2(layout.mode_shape[lm]))
        local_bit_start[lm] = offset
        offset += num_bits

    spatial_bit_start: dict[int, int] = {}
    offset = 0
    for sm in reversed(layout.spatial_modes):
        if sm < 0:
            mode_size = -sm
        else:
            mode_size = layout.mode_shape[sm]
            spatial_bit_start[sm] = offset
        num_bits = int(math.log2(mode_size))
        offset += num_bits

    bits: list[DimBit] = []
    # Innermost mode first (smallest stride = lowest dim-axis bits)
    for mode in reversed(dim_modes):
        mode_extent = layout.mode_shape[mode]
        num_bits = int(math.log2(mode_extent))
        if mode in layout.local_modes:
            start = local_bit_start[mode]
            for b in range(num_bits):
                bits.append(DimBit(level="local", pos=start + b))
        else:
            start = spatial_bit_start[mode]
            for b in range(num_bits):
                spatial_bit = start + b
                if spatial_bit < 5:
                    bits.append(DimBit(level="lane", pos=spatial_bit))
                else:
                    bits.append(DimBit(level="warp", pos=spatial_bit - 5))
    return bits


# ---------------------------------------------------------------------------
# Guard masks
# ---------------------------------------------------------------------------


def _level_masks(bits: list[DimBit], up_to: int) -> tuple[int, int, int]:
    """Return (local_mask, lane_mask, warp_mask) over dim bits [0..up_to]."""
    local_mask = lane_mask = warp_mask = 0
    for b in bits[: up_to + 1]:
        if b.level == "local":
            local_mask |= 1 << b.pos
        elif b.level == "lane":
            lane_mask |= 1 << b.pos
        else:
            warp_mask |= 1 << b.pos
    return local_mask, lane_mask, warp_mask


def _build_guard(
    *,
    local_mask: int,
    lane_mask: int,
    warp_mask: int,
    local_expected: int | None = None,
    lane_expected: int | None = None,
    warp_expected: int | None = None,
    local_idx_expr: Expr,
    lane_id_expr: Expr,
    warp_id_expr: Expr,
) -> Expr:
    """Guard: for each level, ``(coord & mask) == expected`` on the dim bits.

    Default ``expected`` equals the mask, i.e., all dim bits up to and
    including round ``d`` are set. To express "lower bits all set AND this
    bit clear" (the left-child guard in Blelloch down-sweep), pass a
    non-default ``*_expected`` that has this-round's bit cleared.
    """
    if local_expected is None:
        local_expected = local_mask
    if lane_expected is None:
        lane_expected = lane_mask
    if warp_expected is None:
        warp_expected = warp_mask
    parts: list[Expr] = []
    if local_mask:
        parts.append((bitwise_and(local_idx_expr, local_mask)) == local_expected)
    if lane_mask:
        parts.append((bitwise_and(lane_id_expr, lane_mask)) == lane_expected)
    if warp_mask:
        parts.append((bitwise_and(warp_id_expr, warp_mask)) == warp_expected)
    if not parts:
        return boolean.true
    expr = parts[0]
    for p in parts[1:]:
        expr = logical_and(expr, p)
    return expr


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


@register_emitter(ScanInst, target=nvgpu_any)
class ScanInstEmitter(BaseInstEmitter):
    def emit(self, inst: ScanInst) -> None:
        x: RegisterTensor = inst.register_input
        y: RegisterTensor = inst.register_output
        layout = x.layout
        dtype = x.dtype
        op = inst.op
        dim = inst.dim
        exclusive = inst.exclusive

        dim_bits = _classify_dim_bits(layout, dim)
        n_bits = len(dim_bits)
        if n_bits == 0:
            # Scan along a size-1 dim is a no-op (exclusive returns identity, inclusive returns input).
            self._copy_buffer(x, y)
            if exclusive:
                self._fill_with_identity(y, op, dtype)
            return

        x_buf: Var = self.get_or_allocate_var(x)
        y_buf: Var = self.get_or_allocate_var(y)
        input_is_output = x is y

        # For inclusive with input-is-output, we need to save the original input
        # before Blelloch overwrites it.
        scratch_buf: Var | None = None
        if (not exclusive) and input_is_output:
            scratch = RegisterTensor.create(dtype=dtype, shape=x.shape, optional_layout=x.optional_layout)
            scratch_buf = self.get_or_allocate_var(scratch)
            with self.for_range(layout.local_size, attr="u") as i_local:
                self.buffer_store(scratch_buf, indices=[i_local], value=x_buf[i_local])

        # Copy input → output (if they're distinct tensors).
        if not input_is_output:
            with self.for_range(layout.local_size, attr="u") as i_local:
                self.buffer_store(y_buf, indices=[i_local], value=x_buf[i_local])

        lane_id = self.lane_id("scan_lane_id")
        warp_id = self.warp_id("scan_warp_id")

        # Up-sweep: d = 0, 1, ..., n_bits - 1
        for d in range(n_bits):
            self._sweep_round(
                inst_op=op,
                dtype=dtype,
                layout=layout,
                y_buf=y_buf,
                dim_bits=dim_bits,
                round_d=d,
                lane_id=lane_id,
                warp_id=warp_id,
                up_sweep=True,
            )

        # Zero the root (set the "all dim-axis bits set" position to identity).
        self._write_identity_at_full_mask(
            op=op,
            dtype=dtype,
            layout=layout,
            y_buf=y_buf,
            dim_bits=dim_bits,
            lane_id=lane_id,
            warp_id=warp_id,
        )

        # Down-sweep: d = n_bits - 1, ..., 1, 0
        for d in reversed(range(n_bits)):
            self._sweep_round(
                inst_op=op,
                dtype=dtype,
                layout=layout,
                y_buf=y_buf,
                dim_bits=dim_bits,
                round_d=d,
                lane_id=lane_id,
                warp_id=warp_id,
                up_sweep=False,
            )

        # Inclusive: y[i] = y[i] ⊕ original_x[i].
        if not exclusive:
            src_buf = scratch_buf if input_is_output else x_buf
            with self.for_range(layout.local_size, attr="u") as i_local:
                self.buffer_store(
                    y_buf,
                    indices=[i_local],
                    value=_combine(op, y_buf[i_local], src_buf[i_local]),
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _copy_buffer(self, x: RegisterTensor, y: RegisterTensor) -> None:
        if x is y:
            return
        x_buf = self.get_or_allocate_var(x)
        y_buf = self.get_or_allocate_var(y)
        with self.for_range(x.layout.local_size, attr="u") as i_local:
            self.buffer_store(y_buf, indices=[i_local], value=x_buf[i_local])

    def _fill_with_identity(self, y: RegisterTensor, op: str, dtype: DataType) -> None:
        y_buf = self.get_or_allocate_var(y)
        with self.for_range(y.layout.local_size, attr="u") as i_local:
            self.buffer_store(y_buf, indices=[i_local], value=_identity(op, dtype))

    def _write_identity_at_full_mask(
        self,
        *,
        op: str,
        dtype: DataType,
        layout: RegisterLayout,
        y_buf: Var,
        dim_bits: list[DimBit],
        lane_id: Expr,
        warp_id: Expr,
    ) -> None:
        """Set the register slot at the full-dim-bits-set position to the op's identity."""
        local_mask, lane_mask, warp_mask = _level_masks(dim_bits, up_to=len(dim_bits) - 1)
        local_shape = layout.local_shape

        with self.for_range(layout.local_size, attr="u") as i_local:
            local_indices = index_deserialize(i_local, shape=local_shape)
            local_idx_expr = self._local_indices_to_local_idx(layout, local_indices)
            guard = _build_guard(
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                local_idx_expr=local_idx_expr,
                lane_id_expr=lane_id,
                warp_id_expr=warp_id,
            )
            with self.if_then(guard):
                self.buffer_store(y_buf, indices=[i_local], value=_identity(op, dtype))

    def _sweep_round(
        self,
        *,
        inst_op: str,
        dtype: DataType,
        layout: RegisterLayout,
        y_buf: Var,
        dim_bits: list[DimBit],
        round_d: int,
        lane_id: Expr,
        warp_id: Expr,
        up_sweep: bool,
    ) -> None:
        """One round of Blelloch sweep at dim-axis bit ``round_d``.

        Up-sweep: if guard set (bits 0..d all set), y[i] = combine(y[i], y[partner]).
        Down-sweep: if guard set, swap and combine:
            temp = y[partner]
            y[partner] = y[i]
            y[i] = combine(y[i], temp)
        """
        this_bit = dim_bits[round_d]
        local_mask, lane_mask, warp_mask = _level_masks(dim_bits, up_to=round_d)

        if this_bit.level == "local":
            self._round_local(
                inst_op=inst_op,
                dtype=dtype,
                layout=layout,
                y_buf=y_buf,
                partner_delta_local=1 << this_bit.pos,
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                lane_id=lane_id,
                warp_id=warp_id,
                up_sweep=up_sweep,
            )
        elif this_bit.level == "lane":
            self._round_lane(
                inst_op=inst_op,
                dtype=dtype,
                layout=layout,
                y_buf=y_buf,
                lane_bit=this_bit.pos,
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                lane_id=lane_id,
                warp_id=warp_id,
                up_sweep=up_sweep,
            )
        else:
            self._round_warp(
                inst_op=inst_op,
                dtype=dtype,
                layout=layout,
                y_buf=y_buf,
                warp_bit=this_bit.pos,
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                lane_id=lane_id,
                warp_id=warp_id,
                up_sweep=up_sweep,
            )

    # ---- Round implementations ---------------------------------------

    def _round_local(
        self,
        *,
        inst_op: str,
        dtype: DataType,
        layout: RegisterLayout,
        y_buf: Var,
        partner_delta_local: int,  # XOR delta in the local_idx serialization
        local_mask: int,
        lane_mask: int,
        warp_mask: int,
        lane_id: Expr,
        warp_id: Expr,
        up_sweep: bool,
    ) -> None:
        """Combine within a single thread: partner at same thread, different local_idx."""
        local_shape = layout.local_shape
        # Note: local_idx serialization order depends on layout.local_modes. The
        # XOR delta in local_idx flips a bit. We enumerate i_local via
        # for_range, decode to get the local_idx, and combine with partner.
        with self.for_range(layout.local_size, attr="u") as i_local:
            local_indices = index_deserialize(i_local, shape=local_shape)
            local_idx_expr = self._local_indices_to_local_idx(layout, local_indices)
            # Partner slot is local_idx XOR partner_delta_local. Since local_idx
            # serialization is row-major over local_shape, XORing the flat
            # loop index is equivalent.
            i_partner = bitwise_xor(i_local, int32(partner_delta_local))
            guard = _build_guard(
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                local_idx_expr=local_idx_expr,
                lane_id_expr=lane_id,
                warp_id_expr=warp_id,
            )
            with self.if_then(guard):
                if up_sweep:
                    self.buffer_store(
                        y_buf,
                        indices=[i_local],
                        value=_combine(inst_op, y_buf[i_local], y_buf[i_partner]),
                    )
                else:
                    # Down-sweep swap-and-combine.
                    tmp = self.declare_var("scan_tmp", tp=dtype, init=y_buf[i_partner])
                    self.buffer_store(y_buf, indices=[i_partner], value=y_buf[i_local])
                    self.buffer_store(
                        y_buf,
                        indices=[i_local],
                        value=_combine(inst_op, y_buf[i_local], tmp),
                    )

    def _round_lane(
        self,
        *,
        inst_op: str,
        dtype: DataType,
        layout: RegisterLayout,
        y_buf: Var,
        lane_bit: int,
        local_mask: int,
        lane_mask: int,
        warp_mask: int,
        lane_id: Expr,
        warp_id: Expr,
        up_sweep: bool,
    ) -> None:
        """Combine between lanes via shfl_xor_sync.

        The shfl is called unconditionally by every lane (required for
        well-defined semantics with a full warp mask) and its result captured
        into a local variable; the guards then gate the actual updates using
        that pre-snapshotted partner value.
        """
        local_shape = layout.local_shape
        delta = 1 << lane_bit
        with self.for_range(layout.local_size, attr="u") as i_local:
            local_indices = index_deserialize(i_local, shape=local_shape)
            local_idx_expr = self._local_indices_to_local_idx(layout, local_indices)

            # Snapshot: every lane reads its partner before any conditional update.
            partner_val = self.declare_var(
                "scan_partner",
                tp=dtype,
                init=shfl_xor_sync(
                    uint32(0xFFFFFFFF),
                    y_buf[i_local],
                    lane_mask=int32(delta),
                    width=int32(32),
                ),
            )

            right_guard = _build_guard(
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                local_idx_expr=local_idx_expr,
                lane_id_expr=lane_id,
                warp_id_expr=warp_id,
            )
            if up_sweep:
                with self.if_then(right_guard):
                    self.buffer_store(
                        y_buf,
                        indices=[i_local],
                        value=_combine(inst_op, y_buf[i_local], partner_val),
                    )
            else:
                # Down-sweep: right_child_new = right_old ⊕ left_old,
                #             left_child_new  = right_old.
                # Both lanes use the snapshotted partner value, so ordering is fine.
                left_guard = _build_guard(
                    local_mask=local_mask,
                    lane_mask=lane_mask,
                    warp_mask=warp_mask,
                    lane_expected=lane_mask ^ delta,
                    local_idx_expr=local_idx_expr,
                    lane_id_expr=lane_id,
                    warp_id_expr=warp_id,
                )
                with self.if_then(right_guard):
                    self.buffer_store(
                        y_buf,
                        indices=[i_local],
                        value=_combine(inst_op, y_buf[i_local], partner_val),
                    )
                with self.if_then(left_guard):
                    self.buffer_store(y_buf, indices=[i_local], value=partner_val)

    def _round_warp(
        self,
        *,
        inst_op: str,
        dtype: DataType,
        layout: RegisterLayout,
        y_buf: Var,
        warp_bit: int,
        local_mask: int,
        lane_mask: int,
        warp_mask: int,
        lane_id: Expr,
        warp_id: Expr,
        up_sweep: bool,
    ) -> None:
        """Combine between warps via shared memory.

        Each thread writes its current local-buffer slot to shared; sync; threads
        whose guard holds read from the partner warp's slot and combine (up-sweep)
        or swap-and-combine (down-sweep); sync.
        """
        local_shape = layout.local_shape
        local_size = layout.local_size
        spatial_size = layout.spatial_size
        # Shared scratch: one slot per thread per local element.
        # Index: (warp_id * 32 + lane_id) * local_size + i_local
        scratch_size = spatial_size * local_size
        smem_ctx = self.contexts.smem_alloc_ctx
        smem_ptr = smem_ctx.request_shared_workspace(dtype.nbytes * scratch_size)
        smem_buf = self.declare_var(
            "scan_smem",
            tensor_pointer_type(dtype=dtype, shape=[scratch_size]),
            init=cast(smem_ptr, ~dtype),
        )

        delta = 1 << warp_bit
        partner_warp = bitwise_xor(warp_id, int32(delta))

        # Phase A: write my values into shared.
        with self.for_range(local_size, attr="u") as i_local:
            slot = (warp_id * 32 + lane_id) * local_size + i_local
            self.buffer_store(smem_buf, indices=[slot], value=y_buf[i_local])

        self.sync()

        # Phase B: read partner values and combine under guard.
        with self.for_range(local_size, attr="u") as i_local:
            local_indices = index_deserialize(i_local, shape=local_shape)
            local_idx_expr = self._local_indices_to_local_idx(layout, local_indices)
            partner_slot = (partner_warp * 32 + lane_id) * local_size + i_local
            partner_val = smem_buf[partner_slot]

            guard = _build_guard(
                local_mask=local_mask,
                lane_mask=lane_mask,
                warp_mask=warp_mask,
                local_idx_expr=local_idx_expr,
                lane_id_expr=lane_id,
                warp_id_expr=warp_id,
            )

            if up_sweep:
                with self.if_then(guard):
                    self.buffer_store(
                        y_buf,
                        indices=[i_local],
                        value=_combine(inst_op, y_buf[i_local], partner_val),
                    )
            else:
                # Down-sweep: right child combines, left child receives parent's prefix.
                left_guard = _build_guard(
                    local_mask=local_mask,
                    lane_mask=lane_mask,
                    warp_mask=warp_mask,
                    warp_expected=warp_mask ^ delta,
                    local_idx_expr=local_idx_expr,
                    lane_id_expr=lane_id,
                    warp_id_expr=warp_id,
                )
                with self.if_then(guard):
                    self.buffer_store(
                        y_buf,
                        indices=[i_local],
                        value=_combine(inst_op, y_buf[i_local], partner_val),
                    )
                with self.if_then(left_guard):
                    self.buffer_store(y_buf, indices=[i_local], value=partner_val)

        self.sync()

    # ---- Utility -----------------------------------------------------

    @staticmethod
    def _local_indices_to_local_idx(layout: RegisterLayout, local_indices: list[Expr]) -> Expr:
        """Serialize local_indices (per local_mode) back to the flat local_idx.

        Note: ``local_indices`` is already the row-major decomposition of the
        flat ``i_local`` using ``local_shape``; serializing back gives the same
        ``i_local`` numeric. We return this expression so the guard masks can
        compare against it directly.
        """
        return index_serialize(local_indices, shape=layout.local_shape)
