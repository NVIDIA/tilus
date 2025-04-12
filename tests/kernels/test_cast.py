import struct

import pytest
import tilus
import torch
from tilus.extensions.hidet.ir.dtypes.floats_subbyte import FloatSubbyteType
from tilus.utils import dtype_to_torch
from tqdm import tqdm

from hidet.ir.dtypes import bfloat16, float16, float32, int8, int16, int32
from hidet.ir.dtypes.integer_subbyte import IntegerSubbyteType


@pytest.mark.parametrize("src_dtype", [int32, int16, int8, float32, float16, bfloat16])
@pytest.mark.parametrize("dst_dtype", [int32, int16, int8, float32, float16, bfloat16])
@pytest.mark.parametrize("shape", [[1024]])
def test_cast(src_dtype, dst_dtype, shape):
    if src_dtype.is_integer():
        a = tilus.randint(low=-128, high=127, shape=shape, dtype=src_dtype)
    elif src_dtype.is_float():
        a = tilus.rand(shape=shape, dtype=src_dtype)
    else:
        assert False

    b = tilus.kernels.cast(a, dst_dtype)
    b_torch = a.torch().to(dtype_to_torch(dst_dtype))

    torch.testing.assert_close(b.torch(), b_torch)


def cast_fx_to_f32_ref(n: int, dtype: FloatSubbyteType) -> float:
    nbits, exponent_nbits, mantissa_nbits = dtype.nbits, dtype.exponent_nbits, dtype.mantissa_nbits

    sign = (n >> (nbits - 1)) & 1
    exponent = (n >> mantissa_nbits) & ((1 << exponent_nbits) - 1)
    mantissa = n & ((1 << mantissa_nbits) - 1)

    if sign:
        return -cast_fx_to_f32_ref(n ^ (1 << (nbits - 1)), dtype=dtype)

    if exponent == 0:
        return mantissa * 2 ** (1 - mantissa_nbits) * 2 ** (exponent - (2 ** (exponent_nbits - 1) - 1))
    elif exponent == (1 << exponent_nbits) - 1 and nbits == 8:
        if (exponent_nbits, mantissa_nbits) == (5, 2):
            # f8e5m2 supports both inf and NaN
            # - when exponent == 0b11111 and mantissa == 0, it is inf
            # - when exponent == 0b11111 and mantissa != 0, it is NaN
            if mantissa == 0:
                return float("inf")
            else:
                return float("nan")
        elif (exponent_nbits, mantissa_nbits) == (4, 3):
            # f8e4m3 supports only NaN
            # - when exponent == 0b1111 and mantissa == 0b111, it is NaN
            # - when exponent == 0b1111 and mantissa != 0b111, it is treated as normal number
            if mantissa == (1 << mantissa_nbits) - 1:
                return float("nan")
            else:
                return (
                    (mantissa + 2**mantissa_nbits)
                    * 2 ** (-mantissa_nbits)
                    * 2 ** (exponent - (2 ** (exponent_nbits - 1) - 1))
                )
        else:
            raise NotImplementedError()
    else:
        return (
            (mantissa + 2**mantissa_nbits) * 2 ** (-mantissa_nbits) * 2 ** (exponent - (2 ** (exponent_nbits - 1) - 1))
        )


def reinterpret_float32_as_uint32(f: float) -> int:
    return struct.unpack("I", struct.pack("f", f))[0]


def reinterpret_uint32_as_float32(n: int) -> float:
    return struct.unpack("f", struct.pack("I", n))[0]


def cast_f32_to_fx_ref(f: float, dtype: FloatSubbyteType) -> int:
    bits = reinterpret_float32_as_uint32(f)
    nbits = dtype.nbits
    sign = (bits >> 31) & 1
    exponents = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponents == 0xFF:
        raise NotImplementedError("Subbyte floating-point number does not support NaN and inf in our implementation")
    if exponents == 0:
        # the denormalized number in fp32 is too small, all mapped to zero in fp8
        return sign << 7
    # Convert to normalized form with implicit leading bit
    exp_val = exponents - 127
    mantissa = mantissa | 0x800000
    # Rounding preparation
    # 1        8 23
    # . ........ .......................
    # .    ..... ..
    # 1        e m
    e = dtype.exponent_nbits
    m = dtype.mantissa_nbits
    shift = 23 - m
    round_bit = (mantissa >> (shift - 1)) & 1
    sticky_bits = (mantissa & ((1 << (shift - 1)) - 1)) != 0
    lsb = (mantissa >> shift) & 1
    # Perform rounding to nearest, ties to even
    if round_bit and (sticky_bits or lsb):
        mantissa += 1 << shift
        if mantissa & 0x01000000:
            mantissa >>= 1
            exp_val += 1

    # Adjust mantissa
    mantissa >>= shift
    # Adjust exponent for dtype (bias 2^{e-1} - 1)
    exp_val += (1 << (e - 1)) - 1
    if exp_val >= (1 << e):  # exp_val >= 2^(e - 1)
        # Handle overflow
        return (1 << (nbits - 1)) - 1
    elif exp_val <= 0:
        # Handle underflow
        if exp_val >= 1 - m:
            # denormalized number in fp8
            mantissa >>= (-exp_val) + 1
            exp_val = 0
        else:
            # underflow to zero
            return sign << (nbits - 1)
    else:
        mantissa ^= 0x800000
    # Assemble dtype
    sign = (bits >> 31) & 1
    exp = exp_val & ((1 << e) - 1)
    mantissa = mantissa & ((1 << m) - 1)
    return (sign << (e + m)) | (exp << m) | mantissa


@pytest.mark.parametrize(
    "dtype",
    [
        tilus.f7e2m4,
        tilus.f7e5m1,
        tilus.f7e3m3,
        tilus.f6e3m2,
        tilus.f6e4m1,
        tilus.f6e2m3,
        tilus.f5e3m1,
        tilus.f5e2m2,
        tilus.f4e2m1,
        tilus.f3e1m1,
    ],
)
def test_cast_float_subbyte_to_float32(dtype):
    nbits = dtype.nbits
    for n in range(1 << nbits):
        # for n in [120]:
        x = tilus.full(shape=[128], fill_value=n, dtype=getattr(tilus, "u{}".format(nbits))).view(dtype=dtype)
        y = tilus.kernels.cast(x, dtype=tilus.float32)
        y_expected = tilus.full(shape=[128], fill_value=cast_fx_to_f32_ref(n, dtype=dtype), dtype=tilus.float32)
        torch.testing.assert_close(actual=y.torch(), expected=y_expected.torch(), equal_nan=True)


@pytest.mark.parametrize(
    "dtype",
    [
        tilus.f7e2m4,
        tilus.f7e5m1,
        tilus.f7e3m3,
        tilus.f6e3m2,
        tilus.f6e4m1,
        tilus.f6e2m3,
        tilus.f5e3m1,
        tilus.f5e2m2,
        tilus.f4e2m1,
        tilus.f3e1m1,
    ],
)
def test_cast_float32_to_float_subbyte(dtype):
    nbits = dtype.nbits
    # round trip test: float32 -> float_subbyte -> float32 (keep bitwise identical to original)
    for n in tqdm(range(1 << nbits)):
        x = tilus.full(shape=[128], fill_value=n, dtype=getattr(tilus, "u{}".format(nbits))).view(dtype=dtype)
        y = tilus.kernels.cast(x, dtype=tilus.float32)
        y = tilus.kernels.cast(y, dtype=dtype)
        y_expected = x
        torch.testing.assert_close(actual=y.storage, expected=y_expected.storage)

    # test the rounding is correct
    f32_numbers = []
    n = 1000
    for i in range(1000):
        f32_numbers.append(i / n)
    for i in range(1000 + 1):
        f32_numbers.append(i / n * float(dtype.max_value))
    for f32_number in f32_numbers:
        x = tilus.full(shape=[128], fill_value=f32_number, dtype=tilus.float32)
        y = tilus.kernels.cast(x, dtype=dtype)
        y_expected = tilus.full(
            shape=[128], fill_value=cast_f32_to_fx_ref(f32_number, dtype), dtype=getattr(tilus, "u{}".format(nbits))
        ).view(dtype=dtype)
        torch.testing.assert_close(actual=y.storage, expected=y_expected.storage)


def cast_ix_to_i32_ref(n: int, dtype: IntegerSubbyteType) -> int:
    if dtype.signedness():
        nbits = dtype.nbits
        sign_bit = (n >> (nbits - 1)) & 1
        if sign_bit:
            return n - (1 << nbits)
        else:
            return n
    else:
        return n


@pytest.mark.parametrize(
    "dtype",
    [
        tilus.i7,
        tilus.i6,
        tilus.i5,
        tilus.i4,
        tilus.i3,
        tilus.i2,
        tilus.i1,
        tilus.u7,
        tilus.u6,
        tilus.u5,
        tilus.u4,
        tilus.u3,
        tilus.u2,
        tilus.u1,
    ],
)
def test_cast_int_subbyte_to_int32(dtype: IntegerSubbyteType):
    nbits = dtype.nbits
    for n in range(1 << nbits):
        x = tilus.full(shape=[128], fill_value=n, dtype=getattr(tilus, "u{}".format(nbits))).view(dtype=dtype)
        y = tilus.kernels.cast(x, dtype=tilus.int32)
        y_expected = tilus.full(shape=[128], fill_value=cast_ix_to_i32_ref(n, dtype=dtype), dtype=tilus.int32)
        torch.testing.assert_close(actual=y.torch(), expected=y_expected.torch(), equal_nan=True)
