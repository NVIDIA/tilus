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
"""Block-scaled MMA emitter — strict 1:1 with PTX.

Each ``Tcgen05BlockScaledMma{SS,TS}Inst`` lowers to **one** PTX
``tcgen05.mma.cta_group::C.kind::K.block_scale.scale_vec::S`` call. No splitting
along K/M/N, no ``SFA_ID`` rotation loop, no shape autoscaling — all
program-level orchestration (per-inst-K loops, SFA_ID multiplexing) is the
kernel author's responsibility (see lang docs for ``mma_scaled``).
"""

from __future__ import annotations

from tilus.backends.codegen import CodeGenerationFailed
from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.backends.emitters.cuda.tcgen05.smem_desc import SharedMatrixDescriptor
from tilus.hidet.ir.dtypes import (
    float4_e2m1,
    float6_e2m3,
    float6_e3m2,
    float8_e4m3,
    float8_e5m2,
    float16,
    float32,
    uint32,
    uint64,
)
from tilus.hidet.ir.expr import as_expr, logical_or
from tilus.hidet.ir.primitives.cuda.tcgen05 import (
    COLUMN_STRIDE,
    LANE_STRIDE,
    Tcgen05CtaGroupKind,
    Tcgen05MmaBlockSizeKind,
    Tcgen05MmaKind,
    Tcgen05MmaScaleVecKind,
    tcgen05_encode_mxf4_block_scale_inst_descriptor,
    tcgen05_encode_mxf8f6f4_block_scale_inst_descriptor,
    tcgen05_mma_block_scale_with_shared_a,
    tcgen05_mma_block_scale_with_tmem_a,
)
from tilus.hidet.ir.type import DataType
from tilus.ir.instructions.cuda.tcgen05 import (
    Tcgen05BlockScaledMmaSSInst,
    Tcgen05BlockScaledMmaTSInst,
)
from tilus.ir.layout.cuda.tcgen05.smem import canonicalize_shared_layout
from tilus.ir.layout.utils.cute import CuteLayout
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.target import nvgpu_sm100a

# ----------------------------------------------------------------------------
# String → enum lookup tables (the IR carries strings for IR-functor
# friendliness; the emitter resolves them once per inst).
# ----------------------------------------------------------------------------

_KIND_TABLE: dict[str, Tcgen05MmaKind] = {
    "mxf4": Tcgen05MmaKind.MXF4,
    "mxf4nvf4": Tcgen05MmaKind.MXF4NVF4,
    "mxf8f6f4": Tcgen05MmaKind.MXF8F6F4,
}

_SCALE_VEC_TABLE: dict[str, Tcgen05MmaScaleVecKind] = {
    "1X": Tcgen05MmaScaleVecKind.SCALE_VEC_1X,
    "2X": Tcgen05MmaScaleVecKind.SCALE_VEC_2X,
    "4X": Tcgen05MmaScaleVecKind.SCALE_VEC_4X,
}

_BLOCK_SIZE_TABLE: dict[int, Tcgen05MmaBlockSizeKind] = {
    16: Tcgen05MmaBlockSizeKind.BLOCK16,
    32: Tcgen05MmaBlockSizeKind.BLOCK32,
}

# ----------------------------------------------------------------------------
# Inst-desc dtype encodings (PTX 9.7.16.4 Tables 43 & 44).
# ----------------------------------------------------------------------------

# d_dtype field for mxf8f6f4 (Table 43, bits 4-5 are SFB ID, so d_dtype is
# implicitly FP32 = 1 — there's no explicit d field here). For mxf4 (Table 44),
# d is also implicit. We accept FP16 and FP32 per Table 39 and encode 1 for
# FP32, 0 for FP16, but in practice the tables don't expose this knob; mxf8f6f4
# is FP32-only. We'll only set d_dtype=1 (FP32) since both kinds support it.
_F8_F6_F4_DTYPE_CODE: dict[str, int] = {
    # mxf8f6f4 (Table 43): a/b types with 3-bit field
    "f8e4m3": 0,
    "f8e5m2": 1,
    "f6e2m3": 3,
    "f6e3m2": 4,
    "f4e2m1": 5,
}

_MXF4_DTYPE_CODE: dict[str, int] = {
    # mxf4 / mxf4nvf4 (Table 44): a is 3-bit, b is 2-bit; only E2M1 = 1 is legal.
    "f4e2m1": 1,
}

# SF dtype bit (bit 23): 1 = UE8M0, 0 = UE4M3.
_SF_DTYPE_IS_UE8M0: dict[str, int] = {
    "f8e8m0": 1,
    "f8e4m3": 0,
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _check_majorness(
    a_major_kind: str, b_major_kind: str, type_size: int, kind: Tcgen05MmaKind
) -> None:
    """Validate the (A major, B major, type size) combination against PTX Table 54.

    For block-scaled mxf4/mxf4nvf4 kinds, transpose is unsupported (Table 51), so
    both A and B must be K-major.
    """
    if kind in (Tcgen05MmaKind.MXF4, Tcgen05MmaKind.MXF4NVF4):
        if a_major_kind != "K" or b_major_kind != "K":
            raise CodeGenerationFailed(
                f"kind::{kind.value} requires K-major A and B (no transpose), "
                f"got a={a_major_kind}, b={b_major_kind}"
            )
    if a_major_kind == "K" and b_major_kind == "K":
        if type_size not in (4, 6, 8, 16, 32):
            raise CodeGenerationFailed(
                f"Type size {type_size} not supported for K/K majorness."
            )
    elif a_major_kind == "MN" and b_major_kind == "MN":
        if type_size not in (8, 16):
            raise CodeGenerationFailed(
                f"Type size {type_size} not supported for MN/MN majorness."
            )
    else:
        raise CodeGenerationFailed(
            f"Mixed majorness not supported: a={a_major_kind}, b={b_major_kind}"
        )


def _build_inst_desc(
    inst: Tcgen05BlockScaledMmaSSInst | Tcgen05BlockScaledMmaTSInst,
    a_dtype: DataType,
    b_dtype: DataType,
    sfa_dtype: DataType,
    inst_M: int,
    inst_N: int,
    a_major_kind: str,
    b_major_kind: str,
) -> int:
    """Build the 32-bit inst-desc per Table 43 (mxf8f6f4) or Table 44 (mxf4 / mxf4nvf4).

    For ``cta_group::2`` the M field encodes the **logical** (cluster-level)
    M = 2 × per-CTA M.
    """
    # M >> 7: 2-bit field encoding {128, 256} → {1, 2}.
    cluster_M = inst_M * inst.cta_group  # logical (cluster-level) M
    shifted_m_minus_7 = cluster_M >> 7  # 1 or 2
    # N >> 3: 6-bit field encoding N step 8 (we use logical N for cta=2).
    cluster_N = inst_N * inst.cta_group  # logical (cluster-level) N
    shifted_n = cluster_N >> 3
    sf_dtype_code = _SF_DTYPE_IS_UE8M0.get(sfa_dtype.short_name)
    if sf_dtype_code is None:
        raise CodeGenerationFailed(
            f"SF dtype {sfa_dtype.name} not supported for block-scaled MMA"
        )

    if inst.kind == "mxf8f6f4":
        a_code = _F8_F6_F4_DTYPE_CODE.get(a_dtype.short_name)
        b_code = _F8_F6_F4_DTYPE_CODE.get(b_dtype.short_name)
        if a_code is None or b_code is None:
            raise CodeGenerationFailed(
                f"a/b dtype not in mxf8f6f4 table: a={a_dtype.name}, b={b_dtype.name}"
            )
        return tcgen05_encode_mxf8f6f4_block_scale_inst_descriptor(
            sparsity=0,
            d_dtype=1,  # FP32 (mxf8f6f4 is FP32-only per Table 39)
            a_dtype=a_code,
            b_dtype=b_code,
            negate_a=0,
            negate_b=0,
            transpose_a=1 if a_major_kind == "MN" else 0,
            transpose_b=1 if b_major_kind == "MN" else 0,
            shifted_n=shifted_n,
            shifted_m_minus_7=shifted_m_minus_7,
            sf_b_id=inst.sfb_id,
            sf_a_id=inst.sfa_id,
            sf_dtype_is_ue8m0=sf_dtype_code,
        )
    elif inst.kind in ("mxf4", "mxf4nvf4"):
        a_code = _MXF4_DTYPE_CODE.get(a_dtype.short_name)
        b_code = _MXF4_DTYPE_CODE.get(b_dtype.short_name)
        if a_code is None or b_code is None:
            raise CodeGenerationFailed(
                f"a/b dtype not in mxf4 table: a={a_dtype.name}, b={b_dtype.name}"
            )
        return tcgen05_encode_mxf4_block_scale_inst_descriptor(
            sparsity=0,
            d_dtype=1,
            a_dtype=a_code,
            b_dtype=b_code,
            negate_a=0,
            negate_b=0,
            shifted_n=shifted_n,
            shifted_m_minus_7=shifted_m_minus_7,
            sf_b_id=inst.sfb_id,
            sf_a_id=inst.sfa_id,
            sf_dtype_is_ue8m0=sf_dtype_code,
            k_dim_select=0,  # 0 = K=64 dense (we don't support K=96 yet)
        )
    else:
        raise CodeGenerationFailed(f"Unknown kind: {inst.kind}")


# ----------------------------------------------------------------------------
# SS variant (A in shared memory)
# ----------------------------------------------------------------------------


@register_emitter(Tcgen05BlockScaledMmaSSInst, target=nvgpu_sm100a)
class Tcgen05BlockScaledMmaSSEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05BlockScaledMmaSSInst) -> None:
        self.assert_is_warp_aligned(inst, "tcgen05.mma_scaled is a warp-cooperative instruction")
        a_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()
        sfa_tensor: TMemoryTensor = inst.inputs[3].as_tmemory_tensor()
        sfb_tensor: TMemoryTensor = inst.inputs[4].as_tmemory_tensor()

        # Per-CTA shapes (validation is at lang layer, but recompute for inst-desc).
        inst_M, K = a_tensor.shape[0], a_tensor.shape[1]
        inst_N = b_tensor.shape[1]

        # Canonicalize shared layouts for A and B.
        a_canonical = canonicalize_shared_layout(a_tensor.layout, dtype=a_tensor.dtype)
        b_canonical = canonicalize_shared_layout(b_tensor.layout.transpose(), dtype=b_tensor.dtype)
        if a_canonical is None:
            raise CodeGenerationFailed(f"Cannot canonicalize a layout: {a_tensor.layout}")
        if b_canonical is None:
            raise CodeGenerationFailed(f"Cannot canonicalize b layout: {b_tensor.layout}")
        if a_canonical.swizzle_mode != b_canonical.swizzle_mode:
            raise CodeGenerationFailed(
                f"a and b must use the same swizzle mode, got "
                f"{a_canonical.swizzle_mode} vs {b_canonical.swizzle_mode}"
            )
        kind_enum = _KIND_TABLE[inst.kind]
        scale_vec_enum = _SCALE_VEC_TABLE[inst.scale_vec]
        block_size_enum = _BLOCK_SIZE_TABLE[inst.sf_block_size]
        _check_majorness(
            a_canonical.major_kind, b_canonical.major_kind,
            type_size=a_tensor.dtype.nbits, kind=kind_enum,
        )

        # Build matrix descriptors for A and B (shared-memory descriptors).
        # The whole inst (one PTX call) reads the entire A/B per-CTA tile, so
        # the descriptor's start address is the tensor base.
        a_nbits = a_tensor.dtype.nbits
        b_nbits = b_tensor.dtype.nbits
        a_shared_addr = self.shared_tensor_shared_space_addr[a_tensor]
        b_shared_addr = self.shared_tensor_shared_space_addr[b_tensor]
        a_desc_value = SharedMatrixDescriptor(
            addr=a_shared_addr,
            lbo=a_canonical.LBO * a_nbits // 8,
            sbo=a_canonical.SBO * a_nbits // 8,
            base_offset=0, stride_mode=0,
            swizzle_mode=a_canonical.swizzle_mode.encode(),
        )
        b_desc_value = SharedMatrixDescriptor(
            addr=b_shared_addr,
            lbo=b_canonical.LBO * b_nbits // 8,
            sbo=b_canonical.SBO * b_nbits // 8,
            base_offset=0, stride_mode=0,
            swizzle_mode=b_canonical.swizzle_mode.encode(),
        )

        # SF TMEM addresses — these are the TMEM bases of sfa/sfb.
        sfa_tmem_addr = self.tensor2var[sfa_tensor]
        sfb_tmem_addr = self.tensor2var[sfb_tensor]
        d_tmem_addr = self.tensor2var[d_tensor]

        # Build inst-desc.
        idesc_value = _build_inst_desc(
            inst,
            a_dtype=a_tensor.dtype, b_dtype=b_tensor.dtype, sfa_dtype=sfa_tensor.dtype,
            inst_M=inst_M, inst_N=inst_N,
            a_major_kind=a_canonical.major_kind, b_major_kind=b_canonical.major_kind,
        )

        # Emit one PTX call.
        with self.single_thread():
            i_desc = self.declare_var("i_desc", tp=uint32, init=as_expr(idesc_value))
            a_desc = self.declare_var("a_desc", tp=uint64, init=a_desc_value.encoded())
            b_desc = self.declare_var("b_desc", tp=uint64, init=b_desc_value.encoded())
            self.append(
                tcgen05_mma_block_scale_with_shared_a(
                    d_tmem=d_tmem_addr,
                    a_desc=a_desc,
                    b_desc=b_desc,
                    scale_a_tmem=sfa_tmem_addr,
                    scale_b_tmem=sfb_tmem_addr,
                    i_desc=i_desc,
                    enable_input_d=inst.enable_input_d,
                    cta_group=Tcgen05CtaGroupKind.from_int(inst.cta_group),
                    mma_kind=kind_enum,
                    scale_vec=scale_vec_enum,
                    block_size=block_size_enum,
                    predicate=self.contexts.leader_lane_ctx.leader_lane,
                )
            )


# ----------------------------------------------------------------------------
# TS variant (A in TMEM) — same logic, different primitive call.
# ----------------------------------------------------------------------------


@register_emitter(Tcgen05BlockScaledMmaTSInst, target=nvgpu_sm100a)
class Tcgen05BlockScaledMmaTSEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05BlockScaledMmaTSInst) -> None:
        self.assert_is_warp_aligned(inst, "tcgen05.mma_scaled is a warp-cooperative instruction")
        a_tensor: TMemoryTensor = inst.inputs[0].as_tmemory_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()
        sfa_tensor: TMemoryTensor = inst.inputs[3].as_tmemory_tensor()
        sfb_tensor: TMemoryTensor = inst.inputs[4].as_tmemory_tensor()

        inst_M, K = a_tensor.shape[0], a_tensor.shape[1]
        inst_N = b_tensor.shape[1]

        # For TS, A is in TMEM (already canonical); only B has a SMEM swizzle
        # to canonicalize.
        b_canonical = canonicalize_shared_layout(b_tensor.layout.transpose(), dtype=b_tensor.dtype)
        if b_canonical is None:
            raise CodeGenerationFailed(f"Cannot canonicalize b layout: {b_tensor.layout}")
        kind_enum = _KIND_TABLE[inst.kind]
        scale_vec_enum = _SCALE_VEC_TABLE[inst.scale_vec]
        block_size_enum = _BLOCK_SIZE_TABLE[inst.sf_block_size]
        # For TS, a_major_kind is "K" by convention (TMEM A is row-major K-aligned).
        _check_majorness("K", b_canonical.major_kind, type_size=a_tensor.dtype.nbits, kind=kind_enum)

        b_nbits = b_tensor.dtype.nbits
        b_shared_addr = self.shared_tensor_shared_space_addr[b_tensor]
        b_desc_value = SharedMatrixDescriptor(
            addr=b_shared_addr,
            lbo=b_canonical.LBO * b_nbits // 8,
            sbo=b_canonical.SBO * b_nbits // 8,
            base_offset=0, stride_mode=0,
            swizzle_mode=b_canonical.swizzle_mode.encode(),
        )
        a_tmem_addr = self.tensor2var[a_tensor]
        sfa_tmem_addr = self.tensor2var[sfa_tensor]
        sfb_tmem_addr = self.tensor2var[sfb_tensor]
        d_tmem_addr = self.tensor2var[d_tensor]

        idesc_value = _build_inst_desc(
            inst,
            a_dtype=a_tensor.dtype, b_dtype=b_tensor.dtype, sfa_dtype=sfa_tensor.dtype,
            inst_M=inst_M, inst_N=inst_N,
            a_major_kind="K", b_major_kind=b_canonical.major_kind,
        )

        with self.single_thread():
            i_desc = self.declare_var("i_desc", tp=uint32, init=as_expr(idesc_value))
            b_desc = self.declare_var("b_desc", tp=uint64, init=b_desc_value.encoded())
            self.append(
                tcgen05_mma_block_scale_with_tmem_a(
                    d_tmem=d_tmem_addr,
                    a_tmem=a_tmem_addr,
                    b_desc=b_desc,
                    scale_a_tmem=sfa_tmem_addr,
                    scale_b_tmem=sfb_tmem_addr,
                    i_desc=i_desc,
                    enable_input_d=inst.enable_input_d,
                    cta_group=Tcgen05CtaGroupKind.from_int(inst.cta_group),
                    mma_kind=kind_enum,
                    scale_vec=scale_vec_enum,
                    block_size=block_size_enum,
                    predicate=self.contexts.leader_lane_ctx.leader_lane,
                )
            )
