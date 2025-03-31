import pytest
import tilus
import torch
from tilus.utils import dtype_to_torch

from hidet.ir.dtypes import bfloat16, float16, float32, int8, int16, int32


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
