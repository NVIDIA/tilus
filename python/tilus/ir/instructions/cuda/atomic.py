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
"""Tile-level atomic RMW instructions (element-wise and scatter).

The non-atomic scatter-store siblings (``StoreGlobalScatterInst`` /
``StoreSharedScatterInst``) live in :mod:`tilus.ir.instructions.generic` next
to the regular store instructions, since they are plain stores rather than
atomics. The shape-validation helper they share with the atomic scatter
variants is :func:`tilus.ir.instructions.generic.check_scatter_shapes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from tilus.ir.inst import Instruction, InstructionError
from tilus.ir.instructions.generic import check_scatter_shapes
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor

# Allowed RMW opcodes shared by element-wise and scatter atomic instructions.
ATOMIC_OPS: tuple[str, ...] = ("add", "sub", "min", "max", "exch", "cas")
# Allowed PTX memory-ordering semantics.
ATOMIC_SEMS: tuple[str, ...] = ("relaxed", "acquire", "release", "acq_rel")
# Allowed PTX sync scopes.
ATOMIC_SCOPES: tuple[str, ...] = ("cta", "cluster", "gpu", "sys")


def _check_op_sem_scope(op: str, sem: str, scope: str) -> None:
    if op not in ATOMIC_OPS:
        raise InstructionError(f"atomic op must be one of {ATOMIC_OPS}, got {op!r}")
    if sem not in ATOMIC_SEMS:
        raise InstructionError(f"atomic sem must be one of {ATOMIC_SEMS}, got {sem!r}")
    if scope not in ATOMIC_SCOPES:
        raise InstructionError(f"atomic scope must be one of {ATOMIC_SCOPES}, got {scope!r}")


@dataclass(frozen=True, eq=False)
class AtomicSharedInst(Instruction):
    """Element-wise atomic RMW on a shared tensor.

    For each logical element i of ``dst``, performs ``dst[i] = op(dst[i], values[i])``
    atomically. ``dst.shape`` and ``values.shape`` must be equal, and the register
    layout of ``values`` must match ``dst``'s shared layout element-for-element.

    The optional ``output`` register tensor receives the per-element pre-RMW values
    of ``dst``. When ``output`` is ``None`` (set by DCE or the user), codegen emits
    the destination-less PTX form.

    Extra inputs for the ``cas`` opcode (``dst = (dst == compare) ? values : dst``):
    pass ``compare`` as the third input tensor.
    """

    op: str
    sem: str
    scope: str
    VALID_OPS: ClassVar[tuple[str, ...]] = ATOMIC_OPS

    @staticmethod
    def create(
        dst: SharedTensor,
        values: RegisterTensor,
        *,
        op: str,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
        compare: Optional[RegisterTensor] = None,
    ) -> AtomicSharedInst:
        _check_op_sem_scope(op, sem, scope)
        if dst.dtype != values.dtype:
            raise InstructionError(
                f"atomic_shared_{op}: dst dtype {dst.dtype.name} != values dtype {values.dtype.name}"
            )
        if tuple(dst.shape) != tuple(values.shape):
            raise InstructionError(
                f"atomic_shared_{op}: dst shape {list(dst.shape)} != values shape {list(values.shape)}"
            )
        if op == "cas":
            if compare is None:
                raise InstructionError("atomic_shared_cas requires a compare tensor")
            if compare.dtype != dst.dtype or tuple(compare.shape) != tuple(dst.shape):
                raise InstructionError(
                    f"atomic_shared_cas: compare must match dst in dtype/shape, got "
                    f"{compare.dtype.name}{list(compare.shape)} vs {dst.dtype.name}{list(dst.shape)}"
                )
            inputs: tuple = (dst, values, compare)
        else:
            if compare is not None:
                raise InstructionError(f"atomic_shared_{op}: compare is only valid for op='cas'")
            inputs = (dst, values)
        return AtomicSharedInst(output=output, inputs=inputs, op=op, sem=sem, scope=scope)


@dataclass(frozen=True, eq=False)
class AtomicGlobalInst(Instruction):
    """Element-wise atomic RMW on a global tensor. Semantics mirror ``AtomicSharedInst``."""

    op: str
    sem: str
    scope: str
    VALID_OPS: ClassVar[tuple[str, ...]] = ATOMIC_OPS

    @staticmethod
    def create(
        dst: GlobalTensor,
        values: RegisterTensor,
        *,
        op: str,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
        compare: Optional[RegisterTensor] = None,
    ) -> AtomicGlobalInst:
        _check_op_sem_scope(op, sem, scope)
        if dst.dtype != values.dtype:
            raise InstructionError(
                f"atomic_global_{op}: dst dtype {dst.dtype.name} != values dtype {values.dtype.name}"
            )
        if tuple(dst.shape) != tuple(values.shape):
            raise InstructionError(
                f"atomic_global_{op}: dst shape {list(dst.shape)} != values shape {list(values.shape)}"
            )
        if op == "cas":
            if compare is None:
                raise InstructionError("atomic_global_cas requires a compare tensor")
            if compare.dtype != dst.dtype or tuple(compare.shape) != tuple(dst.shape):
                raise InstructionError(
                    f"atomic_global_cas: compare must match dst in dtype/shape, got "
                    f"{compare.dtype.name}{list(compare.shape)} vs {dst.dtype.name}{list(dst.shape)}"
                )
            inputs: tuple = (dst, values, compare)
        else:
            if compare is not None:
                raise InstructionError(f"atomic_global_{op}: compare is only valid for op='cas'")
            inputs = (dst, values)
        return AtomicGlobalInst(output=output, inputs=inputs, op=op, sem=sem, scope=scope)


@dataclass(frozen=True, eq=False)
class AtomicScatterSharedInst(Instruction):
    """Scatter atomic RMW on a shared tensor.

    For each tile element k in ``indices``/``values``, performs
    ``dst[..., indices[k], ...] = op(dst[..., indices[k], ...], values[k])``
    atomically, where ``indices`` picks positions along ``dim``.

    ``indices.shape == values.shape`` strictly, with identical RegisterLayout.
    Non-``dim`` axes of ``dst`` must match the corresponding axes of ``indices``;
    along ``dim``, ``dst.shape[dim]`` can be larger than ``indices.shape[dim]``.
    Out-of-range index values are undefined (no runtime bounds check).

    The optional ``output`` register tensor has shape ``indices.shape`` and
    receives the per-element pre-RMW values at each scattered location. CAS and
    exchange are not supported in the scatter form (semantics under duplicate
    indices are unclear); use the element-wise variant instead.
    """

    op: str
    dim: int
    sem: str
    scope: str
    VALID_OPS: ClassVar[tuple[str, ...]] = ("add", "sub", "min", "max")

    @staticmethod
    def create(
        dst: SharedTensor,
        indices: RegisterTensor,
        values: RegisterTensor,
        *,
        dim: int,
        op: str,
        sem: str = "relaxed",
        scope: str = "cta",
        output: Optional[RegisterTensor] = None,
    ) -> AtomicScatterSharedInst:
        if op not in AtomicScatterSharedInst.VALID_OPS:
            raise InstructionError(
                f"atomic_shared_scatter op must be one of {AtomicScatterSharedInst.VALID_OPS}, got {op!r}"
            )
        if sem not in ATOMIC_SEMS:
            raise InstructionError(f"atomic sem must be one of {ATOMIC_SEMS}, got {sem!r}")
        if scope not in ATOMIC_SCOPES:
            raise InstructionError(f"atomic scope must be one of {ATOMIC_SCOPES}, got {scope!r}")
        check_scatter_shapes(
            "atomic_shared_scatter", dst_shape=dst.shape, dst_dtype=dst.dtype, indices=indices, values=values, dim=dim
        )
        return AtomicScatterSharedInst(
            output=output,
            inputs=(dst, indices, values),
            op=op,
            dim=dim,
            sem=sem,
            scope=scope,
        )


@dataclass(frozen=True, eq=False)
class AtomicScatterGlobalInst(Instruction):
    """Scatter atomic RMW on a global tensor. Semantics mirror ``AtomicScatterSharedInst``."""

    op: str
    dim: int
    sem: str
    scope: str
    VALID_OPS: ClassVar[tuple[str, ...]] = ("add", "sub", "min", "max")

    @staticmethod
    def create(
        dst: GlobalTensor,
        indices: RegisterTensor,
        values: RegisterTensor,
        *,
        dim: int,
        op: str,
        sem: str = "relaxed",
        scope: str = "gpu",
        output: Optional[RegisterTensor] = None,
    ) -> AtomicScatterGlobalInst:
        if op not in AtomicScatterGlobalInst.VALID_OPS:
            raise InstructionError(
                f"atomic_global_scatter op must be one of {AtomicScatterGlobalInst.VALID_OPS}, got {op!r}"
            )
        if sem not in ATOMIC_SEMS:
            raise InstructionError(f"atomic sem must be one of {ATOMIC_SEMS}, got {sem!r}")
        if scope not in ATOMIC_SCOPES:
            raise InstructionError(f"atomic scope must be one of {ATOMIC_SCOPES}, got {scope!r}")
        check_scatter_shapes(
            "atomic_global_scatter", dst_shape=dst.shape, dst_dtype=dst.dtype, indices=indices, values=values, dim=dim
        )
        return AtomicScatterGlobalInst(
            output=output,
            inputs=(dst, indices, values),
            op=op,
            dim=dim,
            sem=sem,
            scope=scope,
        )
