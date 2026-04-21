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
"""Emitters for element-wise and scatter atomic RMW instructions.

Iterates over each thread's local slice of the input register tile,
reconstructs the tile-global indices from the register layout, computes the
destination byte/element offset, and dispatches to the corresponding hidet
``atom.*`` / ``red.*`` primitive.

When the ``output`` register of an atomic instruction is ``None`` (either
passed as such by the user or nulled by the DCE pass), the emitter uses the
destination-less ``red.*`` primitive instead of ``atom.*``.

Supported dtypes in v1: ``int32``.

Non-atomic scatter stores (``StoreSharedScatterInst``/``StoreGlobalScatterInst``)
live next to the other load/store emitters in
:mod:`tilus.backends.emitters.scatter_ldst`.
"""

from __future__ import annotations

from typing import Sequence

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.dtypes import int32
from tilus.hidet.ir.expr import Expr, Var, cast, index_vars
from tilus.hidet.ir.primitives.cuda.atomic_tile import atom_rmw, red_rmw
from tilus.hidet.ir.tools.rewriter import rewrite
from tilus.hidet.ir.type import DataType
from tilus.ir.instructions.cuda.atomic import (
    AtomicGlobalInst,
    AtomicScatterGlobalInst,
    AtomicScatterSharedInst,
    AtomicSharedInst,
)
from tilus.ir.layout import RegisterLayout
from tilus.target import nvgpu_sm70


def _append_atomic(
    emitter: BaseInstEmitter,
    *,
    op: str,
    sem: str,
    scope: str,
    space: str,
    dtype: DataType,
    addr: Expr,
    value: Expr,
    compare: Expr | None,
    old: Var | None,
    old_index: Expr | int | None,
) -> None:
    """Dispatch one element's atomic op to the right hidet primitive.

    ``old`` is the per-thread register buffer receiving the pre-RMW value, and
    ``old_index`` is the local element slot to write into. When ``old`` is
    ``None`` the emitter falls back to the destination-less ``red.*`` form.

    PTX has no ``atom.sub``; we express ``sub`` as ``add`` with a negated
    operand here, so the primitive side only sees ``add``.
    """
    if op == "sub":
        op = "add"
        value = -cast(value, dtype)

    if old is None:
        # Destination-less red.* — also our path for ops (e.g. exch/cas) that
        # require an output is blocked upstream, so here op is in _RED_OPS.
        if op in ("exch", "cas"):
            raise ValueError(f"atom.{op} requires a destination register; output cannot be None")
        emitter.append(red_rmw(op=op, sem=sem, scope=scope, space=space, dtype=dtype, addr=addr, value=value))
    else:
        call = atom_rmw(
            op=op,
            sem=sem,
            scope=scope,
            space=space,
            dtype=dtype,
            addr=addr,
            value=value,
            compare=compare,
        )
        emitter.buffer_store(buf=old, indices=[old_index], value=call)


# ----------------------------------------------------------------------------
# Element-wise atomics
# ----------------------------------------------------------------------------


@register_emitter(AtomicSharedInst, target=nvgpu_sm70)
class AtomicSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: AtomicSharedInst) -> None:
        dst = inst.inputs[0].as_shared_tensor()
        values = inst.inputs[1].as_register_tensor()
        compare = inst.inputs[2].as_register_tensor() if inst.op == "cas" else None

        dtype: DataType = dst.dtype
        layout: RegisterLayout = values.layout
        values_buf: Var = self.get_or_allocate_var(values)
        compare_buf: Var | None = self.get_or_allocate_var(compare) if compare is not None else None
        old_buf: Var | None = self.get_or_allocate_var(inst.register_output) if inst.output is not None else None

        # Shared-space base address as a 32-bit int.
        smem_base = self.declare_var("smem_addr", tp=int32, init=self.shared_tensor_shared_space_addr[dst])

        rank = len(dst.shape)
        shared_axes = index_vars(num_vars=rank)
        byte_offset_template: Expr = dst.layout.byte_offset(*shared_axes, nbytes=dtype.nbytes)

        with self.for_range(layout.local_size, attr="u") as i:
            global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
            rewrite_map = {axis: idx for axis, idx in zip(shared_axes, global_indices)}
            byte_offset = rewrite(byte_offset_template, rewrite_map=rewrite_map)
            addr = smem_base + byte_offset
            _append_atomic(
                self,
                op=inst.op,
                sem=inst.sem,
                scope=inst.scope,
                space="shared",
                dtype=dtype,
                addr=addr,
                value=values_buf[i],
                compare=compare_buf[i] if compare_buf is not None else None,
                old=old_buf,
                old_index=i,
            )


@register_emitter(AtomicGlobalInst, target=nvgpu_sm70)
class AtomicGlobalInstEmitter(BaseInstEmitter):
    def emit(self, inst: AtomicGlobalInst) -> None:
        dst = inst.inputs[0].as_global_tensor()
        values = inst.inputs[1].as_register_tensor()
        compare = inst.inputs[2].as_register_tensor() if inst.op == "cas" else None

        dtype: DataType = dst.dtype
        layout: RegisterLayout = values.layout
        values_buf: Var = self.get_or_allocate_var(values)
        gmem_buf: Var = self.get_or_allocate_var(dst)
        compare_buf: Var | None = self.get_or_allocate_var(compare) if compare is not None else None
        old_buf: Var | None = self.get_or_allocate_var(inst.register_output) if inst.output is not None else None

        rank = len(dst.shape)
        global_axes = index_vars(num_vars=rank)
        offset_template: Expr = dst.layout(*global_axes)

        with self.for_range(layout.local_size, attr="u") as i:
            global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
            rewrite_map = {axis: idx for axis, idx in zip(global_axes, global_indices)}
            offset = rewrite(offset_template, rewrite_map=rewrite_map)
            addr = cast(~gmem_buf[offset], ~dtype)
            _append_atomic(
                self,
                op=inst.op,
                sem=inst.sem,
                scope=inst.scope,
                space="global",
                dtype=dtype,
                addr=addr,
                value=values_buf[i],
                compare=compare_buf[i] if compare_buf is not None else None,
                old=old_buf,
                old_index=i,
            )


# ----------------------------------------------------------------------------
# Scatter atomics
# ----------------------------------------------------------------------------


def _scatter_dst_indices(tile_global_indices: Sequence[Expr], scatter_idx: Expr, dim: int) -> list[Expr]:
    """Replace ``tile_global_indices[dim]`` with ``scatter_idx`` to form dst indices."""
    out = list(tile_global_indices)
    out[dim] = scatter_idx
    return out


@register_emitter(AtomicScatterSharedInst, target=nvgpu_sm70)
class AtomicScatterSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: AtomicScatterSharedInst) -> None:
        dst = inst.inputs[0].as_shared_tensor()
        indices = inst.inputs[1].as_register_tensor()
        values = inst.inputs[2].as_register_tensor()

        dtype: DataType = dst.dtype
        layout: RegisterLayout = indices.layout
        indices_buf: Var = self.get_or_allocate_var(indices)
        values_buf: Var = self.get_or_allocate_var(values)
        old_buf: Var | None = self.get_or_allocate_var(inst.register_output) if inst.output is not None else None

        smem_base = self.declare_var("smem_addr", tp=int32, init=self.shared_tensor_shared_space_addr[dst])

        rank = len(dst.shape)
        shared_axes = index_vars(num_vars=rank)
        byte_offset_template: Expr = dst.layout.byte_offset(*shared_axes, nbytes=dtype.nbytes)

        with self.for_range(layout.local_size, attr="u") as i:
            tile_global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
            scatter_idx = indices_buf[i]
            dst_indices = _scatter_dst_indices(tile_global_indices, scatter_idx, inst.dim)
            rewrite_map = {axis: idx for axis, idx in zip(shared_axes, dst_indices)}
            byte_offset = rewrite(byte_offset_template, rewrite_map=rewrite_map)
            addr = smem_base + byte_offset
            _append_atomic(
                self,
                op=inst.op,
                sem=inst.sem,
                scope=inst.scope,
                space="shared",
                dtype=dtype,
                addr=addr,
                value=values_buf[i],
                compare=None,
                old=old_buf,
                old_index=i,
            )


@register_emitter(AtomicScatterGlobalInst, target=nvgpu_sm70)
class AtomicScatterGlobalInstEmitter(BaseInstEmitter):
    def emit(self, inst: AtomicScatterGlobalInst) -> None:
        dst = inst.inputs[0].as_global_tensor()
        indices = inst.inputs[1].as_register_tensor()
        values = inst.inputs[2].as_register_tensor()

        dtype: DataType = dst.dtype
        layout: RegisterLayout = indices.layout
        indices_buf: Var = self.get_or_allocate_var(indices)
        values_buf: Var = self.get_or_allocate_var(values)
        gmem_buf: Var = self.get_or_allocate_var(dst)
        old_buf: Var | None = self.get_or_allocate_var(inst.register_output) if inst.output is not None else None

        rank = len(dst.shape)
        global_axes = index_vars(num_vars=rank)
        offset_template: Expr = dst.layout(*global_axes)

        with self.for_range(layout.local_size, attr="u") as i:
            tile_global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
            scatter_idx = indices_buf[i]
            dst_indices = _scatter_dst_indices(tile_global_indices, scatter_idx, inst.dim)
            rewrite_map = {axis: idx for axis, idx in zip(global_axes, dst_indices)}
            offset = rewrite(offset_template, rewrite_map=rewrite_map)
            addr = cast(~gmem_buf[offset], ~dtype)
            _append_atomic(
                self,
                op=inst.op,
                sem=inst.sem,
                scope=inst.scope,
                space="global",
                dtype=dtype,
                addr=addr,
                value=values_buf[i],
                compare=None,
                old=old_buf,
                old_index=i,
            )
