# mypy: disable-error-code="no-untyped-def, attr-defined, valid-type"
import functools
from typing import Optional

from hidet.ir.dtypes import DataType, int32
from hidet.ir.type import void_p
from tilus.ir.layout import spatial
from tilus.lang import Script
from tilus.tensor import Tensor, empty
from tilus.utils import cdiv, lcm, prod


class Cast(Script):
    def __init__(self, src_dtype: DataType, dst_dtype: DataType):
        super().__init__()
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        vector = lcm(lcm(src_dtype.nbits, dst_dtype.nbits), 8)
        self.layout = spatial(128).repeat(vector)

    def __call__(self, n: int32, src_ptr: void_p, dst_ptr: void_p):
        self.attrs.warps = 4
        self.attrs.blocks = [cdiv(n, self.layout.size)]

        offset = self.blockIdx.x * self.layout.size
        g_src = self.global_view(ptr=src_ptr, dtype=self.src_dtype, shape=[n])
        r_src = self.load_global(g_src, offsets=[offset], layout=self.layout)
        r_dst = self.cast(r_src, dtype=self.dst_dtype)
        g_dst = self.global_view(ptr=dst_ptr, dtype=self.dst_dtype, shape=[n])
        self.store_global(g_dst, r_dst, offsets=[offset])


@functools.cache
def _cast(src_dtype: DataType, dst_dtype: DataType):
    return Cast(src_dtype, dst_dtype)


def cast(tensor: Tensor, dtype: DataType, *, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        out = empty(shape=tensor.shape, dtype=dtype)
    src_dtype = tensor.dtype
    dst_dtype = dtype
    n = prod(tensor.shape)
    _cast(src_dtype, dst_dtype)(n, tensor.data_ptr(), out.data_ptr())
    return out
