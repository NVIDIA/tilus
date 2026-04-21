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
"""Emitters for non-atomic scatter stores.

``StoreSharedScatterInst`` / ``StoreGlobalScatterInst`` are plain stores — each
lane writes its ``values[k]`` into ``dst[..., indices[k], ...]`` at the tile's
non-scatter global position. Duplicate indices give last-writer-wins; callers
who need well-defined behaviour under duplicates should use the atomic scatter
variants (emitters in :mod:`tilus.backends.emitters.atomic`).

The emitters sit next to the regular load/store emitters rather than under the
atomic namespace, matching the IR placement in :mod:`tilus.ir.instructions.generic`.
"""

from __future__ import annotations

from typing import Sequence

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.expr import Expr, Var, index_vars
from tilus.hidet.ir.tools.rewriter import rewrite
from tilus.ir.instructions import StoreGlobalScatterInst, StoreSharedScatterInst
from tilus.ir.layout import RegisterLayout


def _scatter_dst_indices(tile_global_indices: Sequence[Expr], scatter_idx: Expr, dim: int) -> list[Expr]:
    """Replace ``tile_global_indices[dim]`` with ``scatter_idx`` to form dst indices."""
    out = list(tile_global_indices)
    out[dim] = scatter_idx
    return out


@register_emitter(StoreSharedScatterInst)
class StoreSharedScatterInstEmitter(BaseInstEmitter):
    def emit(self, inst: StoreSharedScatterInst) -> None:
        dst = inst.inputs[0].as_shared_tensor()
        indices = inst.inputs[1].as_register_tensor()
        values = inst.inputs[2].as_register_tensor()

        layout: RegisterLayout = indices.layout
        indices_buf: Var = self.get_or_allocate_var(indices)
        values_buf: Var = self.get_or_allocate_var(values)
        smem_buf: Var = self.get_or_allocate_var(dst)

        rank = len(dst.shape)
        shared_axes = index_vars(num_vars=rank)
        offset_template: Expr = dst.layout(*shared_axes)

        with self.for_range(layout.local_size, attr="u") as i:
            tile_global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
            scatter_idx = indices_buf[i]
            dst_indices = _scatter_dst_indices(tile_global_indices, scatter_idx, inst.dim)
            rewrite_map = {axis: idx for axis, idx in zip(shared_axes, dst_indices)}
            offset = rewrite(offset_template, rewrite_map=rewrite_map)
            self.buffer_store(buf=smem_buf, indices=[offset], value=values_buf[i])


@register_emitter(StoreGlobalScatterInst)
class StoreGlobalScatterInstEmitter(BaseInstEmitter):
    def emit(self, inst: StoreGlobalScatterInst) -> None:
        dst = inst.inputs[0].as_global_tensor()
        indices = inst.inputs[1].as_register_tensor()
        values = inst.inputs[2].as_register_tensor()

        layout: RegisterLayout = indices.layout
        indices_buf: Var = self.get_or_allocate_var(indices)
        values_buf: Var = self.get_or_allocate_var(values)
        gmem_buf: Var = self.get_or_allocate_var(dst)

        rank = len(dst.shape)
        global_axes = index_vars(num_vars=rank)
        offset_template: Expr = dst.layout(*global_axes)

        with self.for_range(layout.local_size, attr="u") as i:
            tile_global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
            scatter_idx = indices_buf[i]
            dst_indices = _scatter_dst_indices(tile_global_indices, scatter_idx, inst.dim)
            rewrite_map = {axis: idx for axis, idx in zip(global_axes, dst_indices)}
            offset = rewrite(offset_template, rewrite_map=rewrite_map)
            self.buffer_store(buf=gmem_buf, indices=[offset], value=values_buf[i])
