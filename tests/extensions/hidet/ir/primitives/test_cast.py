import functools

import pytest
import tabulate
import torch
from tilus.extensions.hidet.ir.dtypes import (
    float4_e2m1,
    float5_e2m2,
    float5_e3m1,
    float6_e2m3,
    float6_e3m2,
    float6_e4m1,
    float7_e2m4,
    float7_e3m3,
    float7_e4m2,
    float7_e5m1,
)
from tilus.extensions.hidet.ir.dtypes.floats_subbyte import FloatSubbyteType

import hidet
from hidet.ir.dtypes import f32, uint8
from hidet.lang import attrs


@functools.cache
def cast_from_f32_kernel(dtype: FloatSubbyteType):
    from tilus.extensions.hidet.ir.primitives.cuda.cast import cast_subbyte_float_from_f32

    with hidet.script_module() as script_module:

        @hidet.script
        def _cast_from_f32(dst: ~uint8, src: ~f32):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            dst[0] = cast_subbyte_float_from_f32(src[0], dtype)

    func = script_module.build()

    return func


@functools.cache
def cast_to_f32_kernel(dtype: FloatSubbyteType):
    from tilus.extensions.hidet.ir.primitives.cuda.cast import cast_subbyte_float_to_f32

    with hidet.script_module() as script_module:

        @hidet.script
        def _cast_to_f32(dst: ~f32, src: ~uint8):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            dst[0] = cast_subbyte_float_to_f32(src[0], dtype)

    func = script_module.build()

    return func


def cast_from_f32(v_f32, dtype: FloatSubbyteType):
    src = torch.full([1], fill_value=v_f32, dtype=torch.float32).cuda()
    dst = torch.empty([1], dtype=torch.uint8).cuda()
    cast_from_f32_kernel(dtype)(dst, src)
    return dst.item()


def cast_to_f32(n, dtype: FloatSubbyteType):
    src = torch.full([1], fill_value=n, dtype=torch.uint8).cuda()
    dst = torch.empty([1], dtype=torch.float32).cuda()
    cast_to_f32_kernel(dtype)(dst, src)
    return dst.item()


def cast_to_f32_ref(n, dtype: FloatSubbyteType):
    exponent_nbits, mantissa_nbits = dtype.exponent_nbits, dtype.mantissa_nbits

    exponent = (n >> mantissa_nbits) & ((1 << exponent_nbits) - 1)
    mantissa = n & ((1 << mantissa_nbits) - 1)

    if exponent == 0:
        return mantissa * 2 ** (1 - mantissa_nbits) * 2 ** (exponent - (2 ** (exponent_nbits - 1) - 1))
    else:
        return (
            (mantissa + 2**mantissa_nbits) * 2 ** (-mantissa_nbits) * 2 ** (exponent - (2 ** (exponent_nbits - 1) - 1))
        )


@pytest.mark.parametrize(
    "dtype",
    [
        float7_e5m1,
        float7_e4m2,
        float7_e3m3,
        float7_e2m4,
        float6_e4m1,
        float6_e3m2,
        float6_e2m3,
        float5_e3m1,
        float5_e2m2,
        float4_e2m1,
    ],
)
def test_cast(dtype: FloatSubbyteType):
    nbits = dtype.nbits

    headers = ["n", "bits", "expected", "actual", "cast_back"]
    rows = []
    fmt = "{" + ":#0{}b".format(nbits + 2) + "}"
    for n in range(2 ** (nbits - 1)):
        expect = cast_to_f32_ref(n, dtype)
        actual = cast_to_f32(n, dtype)
        cast_back_n = cast_from_f32(actual, dtype)
        rows.append(
            [
                n,
                fmt.format(n),
                expect,
                actual,
                cast_back_n,
            ]
        )
        assert actual == expect, (n, actual, expect)
        assert n == cast_back_n, (n, cast_back_n)

    print(dtype.name)
    print(tabulate.tabulate(rows, headers=headers))
