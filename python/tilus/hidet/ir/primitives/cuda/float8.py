# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CUDA FP8 conversion primitives used by vectorized epilogues."""

from typing import no_type_check

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.ir.stmt import BlackBoxStmt
from tilus.hidet.utils import initialize


@initialize()
def register_functions():
    from tilus.hidet.lang import attrs, script  # pylint: disable=import-outside-toplevel
    from tilus.hidet.lang.types import float32, void_p

    bf16_template = r"""
    float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>({}));
    float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>({}));
    __nv_fp8x2_storage_t p0 = __nv_cvt_float2_to_fp8x2(make_float2(f0.x * {}, f0.y * {}), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t p1 = __nv_cvt_float2_to_fp8x2(make_float2(f1.x * {}, f1.y * {}), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint32_t*>({}) = static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16);
    """

    @no_type_check
    @script
    def scale_bf16x4_to_fp8e4m3x4_(d: void_p, ab: void_p, cd: void_p, s0: float32, s1: float32, s2: float32, s3: float32):
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "scale_bf16x4_to_fp8e4m3x4"
        BlackBoxStmt(bf16_template, ab, cd, s0, s1, s2, s3, d)

    for func in [scale_bf16x4_to_fp8e4m3x4_]:
        assert isinstance(func, Function)
        register_primitive_function(name=func.name, func_or_type=func)


def scale_bf16x4_to_fp8e4m3x4(d: Expr, ab: Expr, cd: Expr, s0: Expr, s1: Expr, s2: Expr, s3: Expr) -> Expr:
    return call_primitive_func("scale_bf16x4_to_fp8e4m3x4", args=[d, ab, cd, s0, s1, s2, s3])
