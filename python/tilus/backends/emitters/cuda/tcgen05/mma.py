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
from __future__ import annotations

from dataclasses import dataclass

from hidet.ir.dtypes import bfloat16, float16, float32, int8, int32, tfloat32, uint8, uint32, uint64
from hidet.ir.expr import Expr, Var, as_expr
from hidet.ir.type import DataType

from tilus.backends.codegen import CodeGenerationFailed
from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.backends.emitters.cuda.tcgen05.allocation import COLUMN_STRIDE, LANE_STRIDE
from tilus.backends.emitters.cuda.tcgen05.smem_desc import SharedMatrixDescriptor
from tilus.extensions.hidet.ir.dtypes import float4_e2m1, float6_e2m3, float6_e3m2, float8_e4m3, float8_e5m2
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05CtaGroupKind,
    Tcgen05MmaKind,
    Tcgen05SwizzleMode,
    tcgen05_encode_mma_inst_descriptor,
    tcgen05_mma_with_shared_a,
)
from tilus.ir.instructions.cuda.tcgen05 import (
    Tcgen05MmaSSInst,
)
from tilus.ir.layout.cuda.tcgen05.smem import canonicalize_shared_layout
from tilus.ir.layout.utils.cute import CuteLayout
from tilus.ir.tensor import SharedTensor, TMemoryTensor
from tilus.target import nvgpu_sm100a
from tilus.utils import gcd


@dataclass
class Tcgen05MmaInstDesc:
    """
    See Also: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instuction-desc-kind-tf32-f16-f8f6f4
    """

    sparsity_selector: int
    sparsity: int
    saturate_for_integer: int
    d_dtype: int
    a_dtype: int
    b_dtype: int
    negate_a: int
    negate_b: int
    transpose_a: int
    transpose_b: int
    n: int
    m: int
    maximim_shift_in_ws: int

    def __str__(self) -> str:
        items = [
            f"sparsity_selector: {self.sparsity_selector}",
            f"sparsity: {self.sparsity}",
            f"saturate_for_integer: {self.saturate_for_integer}",
            f"d_dtype: {self.d_dtype}",
            f"a_dtype: {self.a_dtype}",
            f"b_dtype: {self.b_dtype}",
            f"negate_a: {self.negate_a}",
            f"negate_b: {self.negate_b}",
            f"transpose_a: {self.transpose_a}",
            f"transpose_b: {self.transpose_b}",
            f"n: {self.n}",
            f"m: {self.m}",
            f"maximim_shift_in_ws: {self.maximim_shift_in_ws}",
        ]
        return "Tcgen05MmaInstDesc(" + ",\n  ".join(items) + "\n)"

    def encoded(self) -> int:
        return tcgen05_encode_mma_inst_descriptor(
            sparsity_selector=self.sparsity_selector,
            sparsity=self.sparsity,
            saturate_for_integer=self.saturate_for_integer,
            d_dtype=self.d_dtype,
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            negate_a=self.negate_a,
            negate_b=self.negate_b,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
            shifted_n=self.n >> 3,
            shifted_m=self.m >> 4,
            maximim_shift_in_ws=self.maximim_shift_in_ws,
        )

    @staticmethod
    def create(
        mma_kind: Tcgen05MmaKind,
        sparsity_selector: int,
        sparsity: bool,
        saturate_for_integer: bool,
        d_dtype: DataType,
        a_dtype: DataType,
        b_dtype: DataType,
        negate_a: bool,
        negate_b: bool,
        transpose_a: bool,
        transpose_b: bool,
        n: int,
        m: int,
        maximim_shift_in_ws: int,
    ) -> Tcgen05MmaInstDesc:
        assert sparsity_selector in (0, 1, 2, 3)

        operand_dtype_map: dict[DataType, int]
        accumulator_dtype_map: dict[DataType, int]
        if mma_kind == Tcgen05MmaKind.TF32:
            accumulator_dtype_map = {float32: 0}
            operand_dtype_map = {tfloat32: 0}
        elif mma_kind == Tcgen05MmaKind.F16:
            accumulator_dtype_map = {float16: 0, float32: 1}
            operand_dtype_map = {float16: 0, bfloat16: 1}
        elif mma_kind == Tcgen05MmaKind.F8F6F4:
            accumulator_dtype_map = {float16: 0, float32: 1}
            operand_dtype_map = {float8_e4m3: 0, float8_e5m2: 1, float6_e2m3: 3, float6_e3m2: 4, float4_e2m1: 5}
        elif mma_kind == Tcgen05MmaKind.I8:
            accumulator_dtype_map = {int32: 0}
            operand_dtype_map = {uint8: 0, int8: 1}
        else:
            raise NotImplementedError(f"The given MMA kind is not supported yet: {mma_kind}")

        if mma_kind == Tcgen05MmaKind.I8:
            assert not negate_a and not negate_b
        if mma_kind in (Tcgen05MmaKind.TF32, Tcgen05MmaKind.F16, Tcgen05MmaKind.F8F6F4):
            assert saturate_for_integer == 0

        return Tcgen05MmaInstDesc(
            sparsity_selector=sparsity_selector,
            sparsity=1 if sparsity else 0,
            saturate_for_integer=1 if saturate_for_integer else 0,
            d_dtype=accumulator_dtype_map[d_dtype],
            a_dtype=operand_dtype_map[a_dtype],
            b_dtype=operand_dtype_map[b_dtype],
            negate_a=1 if negate_a else 0,
            negate_b=1 if negate_b else 0,
            transpose_a=1 if transpose_a else 0,
            transpose_b=1 if transpose_b else 0,
            n=n,
            m=m,
            maximim_shift_in_ws=maximim_shift_in_ws,
        )


@dataclass
class Tcgen05MmaSSInstMeta:
    kind: Tcgen05MmaKind
    a_desc: SharedMatrixDescriptor
    b_desc: SharedMatrixDescriptor
    d_tmem_addr: Expr
    cta_group: Tcgen05CtaGroupKind
    i_desc: Tcgen05MmaInstDesc

    def emit(self, sb: BaseInstEmitter) -> None:
        i_desc = sb.declare_var("i_desc", tp=uint32, init=as_expr(self.i_desc.encoded()))
        a_desc = sb.declare_var("a_desc", tp=uint64, init=self.a_desc.encoded())
        b_desc = sb.declare_var("b_desc", tp=uint64, init=self.b_desc.encoded())
        # tcgen05.mma has single-thread semantics - only one thread should issue it
        with sb.if_then(sb.current_thread == 0):
            sb.append(
                tcgen05_mma_with_shared_a(
                    d_tmem=self.d_tmem_addr,
                    a_desc=a_desc,
                    b_desc=b_desc,
                    i_desc=i_desc,
                    enable_input_d=True,
                    cta_group=self.cta_group,
                    mma_kind=self.kind,
                )
            )


@register_emitter(Tcgen05MmaSSInst, target=nvgpu_sm100a)
class TMemoryMmaSSEmitter(BaseInstEmitter):
    @staticmethod
    def get_mma_kind(a_dtype: DataType, b_dtype: DataType, d_dtype: DataType) -> Tcgen05MmaKind:
        """
        See Also: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
        """
        if a_dtype == b_dtype == tfloat32 and d_dtype == float32:
            return Tcgen05MmaKind.TF32
        elif all(dtype in (float16, bfloat16) for dtype in (a_dtype, b_dtype)) and d_dtype in (float16, float32):
            return Tcgen05MmaKind.F16
        elif all(
            dtype in (float8_e4m3, float8_e5m2, float6_e2m3, float4_e2m1) for dtype in (a_dtype, b_dtype)
        ) and d_dtype in (float16, float32):
            return Tcgen05MmaKind.F8F6F4
        elif all(dtype in (int8, uint8) for dtype in (a_dtype, b_dtype)) and d_dtype == int32:
            return Tcgen05MmaKind.I8
        else:
            raise CodeGenerationFailed(
                f"Cannot infer the MMA kind from the given data types of a, b, and d: {a_dtype}, {b_dtype}, and {d_dtype}."
            )

    @staticmethod
    def get_inst_mnk(mma_kind: Tcgen05MmaKind, m_size: int, n_size: int, k_size: int) -> tuple[int, int, int]:
        """
        See Also: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes
        """
        if mma_kind == Tcgen05MmaKind.F16:
            if m_size not in (64, 128):
                raise CodeGenerationFailed(f"The given m_size is not supported for F16 MMA kind: {m_size}")
            if n_size % 8 != 0:
                raise CodeGenerationFailed(f"The given n_size is not supported for F16 MMA kind: {n_size}")
            inst_m = m_size
            inst_n = gcd(n_size, 256)
            inst_k = 16
            return inst_m, inst_n, inst_k
        elif mma_kind == Tcgen05MmaKind.F8F6F4:
            if m_size not in (64, 128):
                raise CodeGenerationFailed(f"The given m_size is not supported for F8F6F4 MMA kind: {m_size}")
            if n_size % 8 != 0 or n_size < 8 or n_size > 256:
                raise CodeGenerationFailed(f"The given n_size is not supported for F8F6F4 MMA kind: {n_size}")
            inst_m = m_size
            inst_n = gcd(n_size, 256)
            inst_k = 32
            return inst_m, inst_n, inst_k
        else:
            raise NotImplementedError(f"The given MMA kind is not supported yet: {mma_kind}")

    @staticmethod
    def check_majorness(a_major_kind: str, b_major_kind: str, type_size: int, swizzle_mode: Tcgen05SwizzleMode) -> None:
        """
        See Also: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrices-valid-type-size-majorness-swizzle
        """
        if a_major_kind == "K" and b_major_kind == "K":
            if type_size not in (4, 6, 8, 16, 32):
                raise CodeGenerationFailed(
                    f"The given type size is not supported for row-column majorness: {type_size}"
                )
        elif a_major_kind == "MN" and b_major_kind == "MN":
            if type_size not in (8, 16):
                raise CodeGenerationFailed(
                    f"The given type size is not supported for column-row majorness: {type_size}"
                )
        else:
            raise CodeGenerationFailed(f"The given majorness is not supported: {a_major_kind} and {b_major_kind}.")

    def emit(self, inst: Tcgen05MmaSSInst) -> None:
        a_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: TMemoryTensor = inst.inputs[2].as_tmemory_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        # check the shapes
        assert len(a_shape) == len(b_shape) == len(d_shape) == 2
        assert a_shape[0] == d_shape[0] and a_shape[1] == b_shape[0] and b_shape[1] == d_shape[1]
        m_size, n_size, k_size = a_shape[0], b_shape[1], a_shape[1]
        if m_size != 128:
            raise NotImplementedError("Only support m_size = 128 for now.")

        # canonicalize the layouts
        a_canonical = canonicalize_shared_layout(a_tensor.layout, dtype=a_tensor.dtype)  # [m, k]
        b_canonical = canonicalize_shared_layout(b_tensor.layout.transpose(), dtype=b_tensor.dtype)  # [n, k]
        if a_canonical is None:
            raise CodeGenerationFailed(f"Cannot canonicalize the layout of a tensor: {a_tensor.layout}.")
        if b_canonical is None:
            raise CodeGenerationFailed(f"Cannot canonicalize the layout of b tensor: {b_tensor.layout}.")

        # check majorness
        if a_canonical.swizzle_mode != b_canonical.swizzle_mode:
            raise CodeGenerationFailed(
                f"The swizzle mode of a and b must be the same, but got {a_canonical.swizzle_mode} and {b_canonical.swizzle_mode}."
            )
        self.check_majorness(
            a_canonical.major_kind,
            b_canonical.major_kind,
            type_size=a_tensor.dtype.nbits,
            swizzle_mode=a_canonical.swizzle_mode,
        )

        # get the ptx inst shape
        mma_kind = self.get_mma_kind(a_tensor.dtype, b_tensor.dtype, d_tensor.dtype)
        inst_m, inst_n, inst_k = self.get_inst_mnk(mma_kind, m_size, n_size, k_size)

        repeat_m = m_size // inst_m
        repeat_n = n_size // inst_n
        repeat_k = k_size // inst_k

        # construct the i_dest
        i_dest = Tcgen05MmaInstDesc.create(
            mma_kind=mma_kind,
            sparsity_selector=0,
            sparsity=False,
            saturate_for_integer=False,
            d_dtype=d_tensor.dtype,
            a_dtype=a_tensor.dtype,
            b_dtype=b_tensor.dtype,
            negate_a=False,
            negate_b=False,
            transpose_a=a_canonical.major_kind == "MN",
            transpose_b=b_canonical.major_kind == "MN",
            n=inst_n,
            m=inst_m,
            maximim_shift_in_ws=0,
        )

        a_cute_layout: CuteLayout = a_canonical.swizzled_cute_layout.layout
        b_cute_layout: CuteLayout = b_canonical.swizzled_cute_layout.layout

        a_shared_addr: Var = self.shared_tensor_shared_space_addr[a_tensor]
        b_shared_addr: Var = self.shared_tensor_shared_space_addr[b_tensor]
        d_tmem_addr: Var = self.tensor2var[d_tensor]

        for k in range(repeat_k):
            for i in range(repeat_m):
                for j in range(repeat_n):
                    # construct the a_dest
                    a_offset = a_cute_layout(i * inst_m, k * inst_k)
                    b_offset = b_cute_layout(j * inst_n, k * inst_k)
                    a_desc = SharedMatrixDescriptor(
                        addr=a_shared_addr + a_offset * a_tensor.dtype.nbytes,
                        lbo=a_canonical.LBO * a_tensor.dtype.nbytes,
                        sbo=a_canonical.SBO * a_tensor.dtype.nbytes,
                        base_offset=0,
                        stride_mode=0,
                        swizzle_mode=a_canonical.swizzle_mode.encode(),
                    )
                    b_desc = SharedMatrixDescriptor(
                        addr=b_shared_addr + b_offset * b_tensor.dtype.nbytes,
                        lbo=b_canonical.LBO * b_tensor.dtype.nbytes,
                        sbo=b_canonical.SBO * b_tensor.dtype.nbytes,
                        base_offset=0,
                        stride_mode=0,
                        swizzle_mode=b_canonical.swizzle_mode.encode(),
                    )
                    d_offset = i * inst_m * LANE_STRIDE + j * inst_n * COLUMN_STRIDE
                    inst_meta = Tcgen05MmaSSInstMeta(
                        kind=mma_kind,
                        a_desc=a_desc,
                        b_desc=b_desc,
                        d_tmem_addr=d_tmem_addr + d_offset,
                        cta_group=Tcgen05CtaGroupKind.CTA_1,
                        i_desc=i_dest,
                    )
                    inst_meta.emit(self)
