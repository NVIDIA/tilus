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
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from tilus.hidet.ir.expr import Constant, Expr
from tilus.hidet.ir.type import DataType
from tilus.ir.inst import Instruction, InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor


@dataclass(frozen=True, eq=False)
class Tcgen05AllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(dtype: DataType, shape: Sequence[int], cta_group: int) -> Tcgen05AllocInst:
        assert len(shape) >= 2, "Tcgen05AllocInst only supports tensors with rank >= 2."
        assert shape[0] in (32, 64, 128), "The first (lane) dimension must be 32, 64, or 128."
        output = TMemoryTensor.create(dtype=dtype, shape=shape)
        return Tcgen05AllocInst(output=output, inputs=(), cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class Tcgen05DeallocInst(Instruction):
    @staticmethod
    def create(tmt: TMemoryTensor) -> Tcgen05DeallocInst:
        return Tcgen05DeallocInst(output=None, inputs=(tmt,))


@dataclass(frozen=True, eq=False)
class Tcgen05RelinquishAllocPermitInst(Instruction):
    cta_group: int = 1

    @staticmethod
    def create(cta_group: int) -> Tcgen05RelinquishAllocPermitInst:
        return Tcgen05RelinquishAllocPermitInst(output=None, inputs=(), cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class Tcgen05SliceInst(Instruction):
    offsets: tuple[Expr, ...]
    slice_dims: tuple[int, ...]

    @staticmethod
    def create(
        tmem: TMemoryTensor,
        offsets: Sequence[Expr],
        slice_dims: Sequence[int],
        slice_shape: Sequence[int],
    ) -> Tcgen05SliceInst:
        assert len(tmem.shape) == len(offsets)
        assert len(slice_shape) == len(slice_dims)
        # The lane dim (0) and the innermost column dim (-1) must always be in the slice.
        assert len(slice_dims) >= 2 and 0 in slice_dims and (len(tmem.shape) - 1) in slice_dims, (
            "The lane dim (0) and the innermost column dim (-1) must be included in the slice."
        )
        assert isinstance(offsets[0], Constant), "The lane (row) offset must be a constant."
        output = TMemoryTensor.create(dtype=tmem.dtype, shape=slice_shape)
        return Tcgen05SliceInst(output=output, inputs=(tmem,), offsets=tuple(offsets), slice_dims=tuple(slice_dims))


@dataclass(frozen=True, eq=False)
class Tcgen05ViewInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> Tcgen05ViewInst:
        if len(tmem.shape) != len(shape):
            raise ValueError("The rank of the new shape must match the original shape.")
        if any(s1 != s2 for s1, s2 in zip(tmem.shape[:-1], shape[:-1])):
            raise ValueError("All dimensions except the last one must match in the view operation.")
        if tmem.shape[-1] * tmem.dtype.nbits != shape[-1] * dtype.nbits:
            raise ValueError(
                "The total number of bits in the last dimension must remain the same in the view operation."
            )
        output = TMemoryTensor.create(dtype=dtype, shape=shape)
        return Tcgen05ViewInst(output=output, inputs=(tmem,))


@dataclass(frozen=True, eq=False)
class Tcgen05LoadInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor) -> Tcgen05LoadInst:
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.load)
        output = RegisterTensor.create(dtype=tmem.dtype, shape=tmem.shape)
        return Tcgen05LoadInst(output=output, inputs=(tmem,))


@dataclass(frozen=True, eq=False)
class Tcgen05StoreInst(Instruction):
    @staticmethod
    def create(tmem: TMemoryTensor, src: RegisterTensor) -> Tcgen05StoreInst:
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.store)
        return Tcgen05StoreInst(output=None, inputs=(tmem, src))


@dataclass(frozen=True, eq=False)
class Tcgen05WaitInst(Instruction):
    wait_load: bool
    wait_store: bool

    @staticmethod
    def create(wait_load: bool, wait_store: bool) -> Tcgen05WaitInst:
        return Tcgen05WaitInst(output=None, inputs=(), wait_load=wait_load, wait_store=wait_store)


@dataclass(frozen=True, eq=False)
class Tcgen05CopyInst(Instruction):
    # Multicast pattern as a primitive string ("" for no multicast, "warpx4",
    # "warpx2_02_13", "warpx2_01_23"). Converted to Tcgen05CopyMulticastKind in
    # the codegen emitter — primitive types are kept here so IR functors can
    # walk the field generically.
    multicast: str = ""

    @staticmethod
    def create(src: SharedTensor, dst: TMemoryTensor, multicast: str = "") -> Tcgen05CopyInst:
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.copy)
        return Tcgen05CopyInst(output=None, inputs=(dst, src), multicast=multicast)


@dataclass(frozen=True, eq=False)
class Tcgen05CommitInst(Instruction):
    mbarrier: Expr
    cta_group: int
    multicast_mask: Optional[int]

    @staticmethod
    def create(mbarrier: Expr, cta_group: int, multicast_mask: Optional[int] = None) -> Tcgen05CommitInst:
        assert cta_group in (1, 2), "cta_group must be 1 or 2, got {}".format(cta_group)
        return Tcgen05CommitInst(
            output=None, inputs=(), mbarrier=mbarrier, cta_group=cta_group, multicast_mask=multicast_mask
        )


@dataclass(frozen=True, eq=False)
class Tcgen05MmaSSInst(Instruction):
    enable_input_d: Expr
    cta_group: int

    @staticmethod
    def create(
        a: SharedTensor,
        b: SharedTensor,
        d: TMemoryTensor,
        enable_input_d: Expr,
        cta_group: int,
    ) -> Tcgen05MmaSSInst:
        if cta_group not in (1, 2):
            raise InstructionError("cta_group must be 1 or 2, got {}".format(cta_group))
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.mma)
        return Tcgen05MmaSSInst(output=None, inputs=(a, b, d), enable_input_d=enable_input_d, cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class Tcgen05MmaTSInst(Instruction):
    enable_input_d: Expr
    cta_group: int

    @staticmethod
    def create(
        a: TMemoryTensor,
        b: SharedTensor,
        d: TMemoryTensor,
        enable_input_d: Expr,
        cta_group: int,
    ) -> Tcgen05MmaTSInst:
        if cta_group not in (1, 2):
            raise InstructionError("cta_group must be 1 or 2, got {}".format(cta_group))
        # Note: 2D validation is performed at the lang layer (Tcgen05InstructionGroup.mma)
        return Tcgen05MmaTSInst(output=None, inputs=(a, b, d), enable_input_d=enable_input_d, cta_group=cta_group)


# ----------------------------------------------------------------------------
# Block-scaled MMA — kept as separate instructions from the unscaled MMA so
# their codegen and validation can stay focused. Each Tcgen05BlockScaledMma*Inst
# lowers to exactly **one** PTX `tcgen05.mma.block_scale` call. All choice of
# (kind, scale_vec, sf_block_size, sfa_id, sfb_id) is determined at the lang
# layer from the operand dtypes + shape, so the IR carries a fully-resolved
# instruction descriptor.
# ----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Tcgen05BlockScaledMmaSSInst(Instruction):
    """One PTX ``tcgen05.mma.cta_group::C.kind::K.block_scale.scale_vec::S`` with shared-A.

    Inputs: ``(a, b, d, sfa, sfb)``. Field meanings:

    * ``kind``: one of ``"mxf8f6f4"``, ``"mxf4"``, ``"mxf4nvf4"`` — the PTX
      ``.kind::*`` modifier.
    * ``scale_vec``: ``"1X"``, ``"2X"``, or ``"4X"`` — the PTX
      ``.scale_vec::NX`` modifier (``"4X"`` is also spelled ``.block16``,
      ``"1X"``/``"2X"`` are ``.block32``).
    * ``sf_block_size``: 16 or 32 — the SF group size in K-elements (kept for
      provenance / introspection; the PTX-level inst is uniquely determined by
      ``scale_vec`` + ``kind``).
    * ``sfa_id``, ``sfb_id``: byte / half-word offset within each TMEM cell
      of SFA/SFB indicating which inst-K iter of a packed cell-round is
      consumed by this MMA. Valid set depends on ``scale_vec`` (4X→{0},
      2X→{0,2}, 1X→{0,1,2,3}).
    * ``cta_group``: 1 or 2 (the ``.cta_group::C`` modifier).
    * ``enable_input_d``: predicate; ``D = A*B + D`` if true, ``D = A*B``
      otherwise.
    """

    kind: str
    scale_vec: str
    sf_block_size: int
    sfa_id: int
    sfb_id: int
    cta_group: int
    enable_input_d: Expr

    @staticmethod
    def create(
        a: SharedTensor,
        b: SharedTensor,
        d: TMemoryTensor,
        sfa: TMemoryTensor,
        sfb: TMemoryTensor,
        kind: str,
        scale_vec: str,
        sf_block_size: int,
        sfa_id: int,
        sfb_id: int,
        cta_group: int,
        enable_input_d: Expr,
    ) -> Tcgen05BlockScaledMmaSSInst:
        if cta_group not in (1, 2):
            raise InstructionError("cta_group must be 1 or 2, got {}".format(cta_group))
        # Validation of (kind, scale_vec, sf_block_size, sfa/sfb shapes & ids)
        # is performed at the lang layer (Tcgen05InstructionGroup.mma_scaled)
        # so that user-facing error messages can reference the support matrix.
        return Tcgen05BlockScaledMmaSSInst(
            output=None,
            inputs=(a, b, d, sfa, sfb),
            kind=kind,
            scale_vec=scale_vec,
            sf_block_size=sf_block_size,
            sfa_id=sfa_id,
            sfb_id=sfb_id,
            cta_group=cta_group,
            enable_input_d=enable_input_d,
        )


@dataclass(frozen=True, eq=False)
class Tcgen05BlockScaledMmaTSInst(Instruction):
    """Block-scaled MMA with A operand in TMEM. See :class:`Tcgen05BlockScaledMmaSSInst`."""

    kind: str
    scale_vec: str
    sf_block_size: int
    sfa_id: int
    sfb_id: int
    cta_group: int
    enable_input_d: Expr

    @staticmethod
    def create(
        a: TMemoryTensor,
        b: SharedTensor,
        d: TMemoryTensor,
        sfa: TMemoryTensor,
        sfb: TMemoryTensor,
        kind: str,
        scale_vec: str,
        sf_block_size: int,
        sfa_id: int,
        sfb_id: int,
        cta_group: int,
        enable_input_d: Expr,
    ) -> Tcgen05BlockScaledMmaTSInst:
        if cta_group not in (1, 2):
            raise InstructionError("cta_group must be 1 or 2, got {}".format(cta_group))
        return Tcgen05BlockScaledMmaTSInst(
            output=None,
            inputs=(a, b, d, sfa, sfb),
            kind=kind,
            scale_vec=scale_vec,
            sf_block_size=sf_block_size,
            sfa_id=sfa_id,
            sfb_id=sfb_id,
            cta_group=cta_group,
            enable_input_d=enable_input_d,
        )
