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
# mypy: disable-error-code="no-untyped-def, attr-defined, valid-type"
import functools
from typing import Optional

from hidet.ir.dtypes import DataType, int32
from hidet.ir.type import void_p

from tilus.ir.layout.ops import spatial
from tilus.lang import Script
from tilus.tensor import Tensor, empty
from tilus.utils import cdiv, lcm, prod


class Cast(Script):
    def __init__(self, src_dtype: DataType, dst_dtype: DataType):
        super().__init__()
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        if src_dtype.nbits < 8 and dst_dtype.nbits < 8:
            raise NotImplementedError()
        elif src_dtype.nbits < 8:
            vector = lcm(src_dtype.nbits, 8) // src_dtype.nbits
        elif dst_dtype.nbits < 8:
            vector = lcm(dst_dtype.nbits, 8) // dst_dtype.nbits
        else:
            bits = lcm(src_dtype.nbits, dst_dtype.nbits)
            assert 128 % bits == 0
            vector = 128 // bits
        self.layout = spatial(128).local(vector)

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
