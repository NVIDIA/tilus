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
from typing import Dict, Tuple

from hidet.ir.dtypes import bfloat16, float16, int8, uint8, uint32
from hidet.ir.dtypes.floats import FloatType
from hidet.ir.expr import Expr, Var, cast, tensor_pointer_var, tensor_var, var
from hidet.ir.primitives.cuda.half import fma_f16x2, sub_f16x2
from hidet.ir.primitives.cuda.prmt import prmt
from hidet.ir.type import Callable, DataType, PointerType, TensorPointerType, TensorType

from tilus.backends.codegen import BaseInstEmitter, Codegen, register_emitter
from tilus.extensions.hidet.ir.dtypes import (
    float6_e3m2,
    float8_e4m3,
    int4b,
    uint1b,
    uint2b,
    uint3b,
    uint4b,
    uint5b,
    uint6b,
    uint7b,
)
from tilus.extensions.hidet.ir.dtypes.floats_subbyte import FloatSubbyteType
from tilus.extensions.hidet.ir.primitives.cuda.bfloat16 import mul_bf16x2
from tilus.extensions.hidet.ir.primitives.cuda.half import mul_f16x2
from tilus.extensions.hidet.ir.primitives.cuda.lop3 import lop3
from tilus.ir.instructions import CastInst
from tilus.target import amdgpu_any, nvgpu_any
from tilus.utils import cdiv


def get_base_type(tp):
    if isinstance(tp, TensorPointerType):
        return tp.tensor_type.dtype
    elif isinstance(tp, TensorType):
        return tp.dtype
    elif isinstance(tp, PointerType):
        return tp.base_type
    else:
        raise ValueError()


class CastInstBaseEmitter(BaseInstEmitter):
    def __init__(self, codegen: Codegen) -> None:
        super().__init__(codegen)
        self.specialized_cast: Dict[Tuple[DataType, DataType], Callable[[Var, Var], None]] = {}
        self.size: int = -1

    def emit(self, inst: CastInst) -> None:  # type: ignore
        src = inst.inputs[0]
        dst = inst.register_output
        self.size = dst.local_size

        var: Var = self.declare(
            tensor_var("casted_{}".format(dst.dtype.short_name), shape=[dst.local_size], dtype=dst.dtype)
        )
        self.tensor2var[dst] = var

        src_var = self.tensor2var[src]
        dst_var = self.tensor2var[dst]

        src_dtype = src.dtype
        dst_dtype = dst.dtype

        if (src_dtype, dst_dtype) in self.specialized_cast:
            impl = self.specialized_cast[(src_dtype, dst_dtype)]
        else:
            impl = self.cast_generic

        impl(src_var, dst_var)

    def cast_generic(self, src: Var, dst: Var) -> None:
        with self.for_range(extent=self.size) as i:
            self.buffer_store(buf=dst, indices=[i], value=src[i])  # implicit cast


@register_emitter(CastInst, target=nvgpu_any)
class NvgpuCastInstEmitter(CastInstBaseEmitter):
    def __init__(self, codegen):
        super().__init__(codegen)
        self.specialized_cast.update(
            {
                (uint3b, float16): self.cast_ux_to_f16,
                (uint5b, float16): self.cast_ux_to_f16,
                (uint6b, float16): self.cast_ux_to_f16,
                (uint7b, float16): self.cast_ux_to_f16,
                (uint1b, float16): self.cast_ux_to_f16,
                (uint2b, float16): self.cast_ux_to_f16,
                (uint4b, float16): self.cast_u4_to_f16,
                (uint8, float16): self.cast_u8_to_f16,
                (int8, float16): self.cast_i8_to_f16,
                (int4b, float16): self.cast_i4_to_f16,
                (float8_e4m3, float16): self.cast_f8e4m3_to_f16,
                (float8_e4m3, bfloat16): self.cast_f8e4m3_to_bf16,
                (float6_e3m2, bfloat16): self.cast_f6e3m2_to_bf16,
            }
        )

    def cast_u8_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 4 != 0:
            self.cast_generic(src, dst)
            return

        src_uint32 = self.declare(
            v=tensor_pointer_var("src_uint32", shape=[self.size // 4], dtype="uint32"), init=cast(src, ~uint32)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        with self.for_range(extent=self.size // 4) as i:
            # cast 4 uint8 to 4 float16 at each time
            # range for uint8: 0 ~ 255

            # transform int8[a, b, c, d] to float16[a, b, c, d] + 1024
            self.append(prmt(d=~dst_uint32[i << 1], a=src_uint32[i], b=uint32(0x00000064), c=uint32(0x4140)))
            self.append(prmt(d=~dst_uint32[i << 1 | 1], a=src_uint32[i], b=uint32(0x00000064), c=uint32(0x4342)))

            # transform float16[a, b, c, d] + 1024 to float16[a, b, c, d]
            # where 0x6400 = 0b0110010000000000 = 1024
            self.append(sub_f16x2(d=~dst_uint32[i << 1], a=dst_uint32[i << 1], b=uint32(0x64006400)))
            self.append(sub_f16x2(d=~dst_uint32[i << 1 | 1], a=dst_uint32[i << 1 | 1], b=uint32(0x64006400)))

    def cast_i8_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 4 != 0:
            self.cast_generic(src, dst)
            return

        src_uint32 = self.declare(
            v=tensor_pointer_var("src_uint32", shape=[self.size // 4], dtype="uint32"), init=cast(src, ~uint32)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        with self.for_range(extent=self.size // 4) as i:
            # cast 4 int8 to 4 float16 at each time
            # range for int8: -128 ~ 127, uint8: 0 ~ 255

            # transform int8[a, b, c, d] to uint8[a, b, c, d] + 128
            self.buffer_store(buf=src_uint32, indices=[i], value=src_uint32[i] ^ uint32(0x80808080))

            # transform int8[a, b, c, d] + 128 to float16[a, b, c, d] + (1024 + 128)
            self.append(prmt(d=~dst_uint32[i << 1], a=src_uint32[i], b=uint32(0x00000064), c=uint32(0x4140)))
            self.append(prmt(d=~dst_uint32[i << 1 | 1], a=src_uint32[i], b=uint32(0x00000064), c=uint32(0x4342)))

            # transform float16[a, b, c, d] + (1024 + 128) to float16[a, b, c, d]
            # where 0x6480 = 0b0110010010000000 = 1152
            self.append(sub_f16x2(d=~dst_uint32[i << 1], a=dst_uint32[i << 1], b=uint32(0x64806480)))
            self.append(sub_f16x2(d=~dst_uint32[i << 1 | 1], a=dst_uint32[i << 1 | 1], b=uint32(0x64806480)))

    def cast_i4_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 8 != 0:
            self.cast_generic(src, dst)
            return

        src_uint32 = self.declare(
            v=tensor_pointer_var("src_uint32", shape=[self.size // 8], dtype="uint32"), init=cast(src, ~uint32)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        with self.for_range(extent=self.size // 8) as i:
            # cast 8 int4b to float16 at each time
            # range for int4b: -8 ~ 7

            # transform int4b[7, 6, 5, 4, 3, 2, 1, 0] to uint4b[7, 6, 5, 4, 3, 2, 1, 0] + 8
            self.buffer_store(buf=src_uint32, indices=[i], value=src_uint32[i] ^ uint32(0x88888888))

            # transform int4b[7, 6, 5, 4, 3, 2, 1, 0] + 8 to float16 [4, 0], [5, 1], [6, 2], [7, 3]
            imm_lut = (0xF0 & 0xCC) | 0xAA
            a = self.declare(v=Var("a", type=uint32), init=src_uint32[i])
            b = self.declare(v=Var("b", type=uint32), init=src_uint32[i] >> 8)
            h = self.declare(v=tensor_var("h", shape=[4], dtype=uint32))
            self.append(
                lop3(~h[0], a, uint32(0x000F000F), uint32(0x64006400), imm_lut=imm_lut)
            )  # (fp16[4, 0] + 8) + 1024
            self.append(
                lop3(~h[1], a, uint32(0x00F000F0), uint32(0x64006400), imm_lut=imm_lut)
            )  # (fp16[5, 1] + 8) * 16 + 1024
            self.append(
                lop3(~h[2], b, uint32(0x000F000F), uint32(0x64006400), imm_lut=imm_lut)
            )  # (fp16[6, 2] + 8) + 1024
            self.append(
                lop3(~h[3], b, uint32(0x00F000F0), uint32(0x64006400), imm_lut=imm_lut)
            )  # (fp16[7, 3] + 8) * 16 + 1024
            self.append(sub_f16x2(d=~h[0], a=h[0], b=uint32(0x64086408)))  # x-1032 => fp16[4, 0]
            self.append(sub_f16x2(d=~h[2], a=h[2], b=uint32(0x64086408)))  # x-1032 => fp16[6, 2]
            self.append(
                fma_f16x2(d=~h[1], a=h[1], b=uint32(0x2C002C00), c=uint32(0xD480D480))
            )  # (x / 16) - 72 => fp16[5, 1]
            self.append(
                fma_f16x2(d=~h[3], a=h[3], b=uint32(0x2C002C00), c=uint32(0xD480D480))
            )  # (x / 16) - 72 => fp16[7, 3]
            self.append(prmt(d=~dst_uint32[i << 2 | 0], a=h[0], b=h[1], c=uint32(0x5410)))  # fp16[1, 0]
            self.append(prmt(d=~dst_uint32[i << 2 | 1], a=h[2], b=h[3], c=uint32(0x5410)))  # fp16[3, 2]
            self.append(prmt(d=~dst_uint32[i << 2 | 2], a=h[0], b=h[1], c=uint32(0x7632)))  # fp16[5, 4]
            self.append(prmt(d=~dst_uint32[i << 2 | 3], a=h[2], b=h[3], c=uint32(0x7632)))  # fp16[7, 6]

    def extract_bits_to_uint32(self, src_uint8: Expr, i: int, size: int, nbits: int, pos: str) -> Var:
        """
        Extract the bits of src[2 * i] and src[2 * i + 1] to uint32 variable. The src_uint8 stored size elements of a
        type with nbits number of bits (nbits <= 8).

        When pos == 'low':
            p[nbits - 1 : 0] = src[2 * i]
            p[nbits + 15 : 16] = src[2 * i + 1]
        When pos == 'high':
            p[31: 32 - nbits] = src[2 * i]
            p[15: 16 - nbits] = src[2 * i + 1]
        """
        p = self.declare_var("p", uint32, init=uint32(0))

        def dst_start_bit_in_uint32(i, j):
            if pos == "low":
                return 16 * j
            elif pos == "high":
                return 16 * j + 16 - nbits
            else:
                raise ValueError(pos)

        # move the src[2 * i] and src[2 * i + 1] to p[...:0] and p[...:16], respectively
        for j in range(2):
            dst_start_bit = dst_start_bit_in_uint32(i, j)

            start_bit = (2 * i + j) * nbits
            end_bit = start_bit + nbits - 1
            if end_bit // 32 * 32 < size * nbits // 32:
                # use uint32 to extract
                if start_bit // 32 == end_bit // 32:
                    # in a single uint32
                    r = self.declare_var("r", uint32, init=cast(src_uint8, ~uint32)[start_bit // 32])
                    self.update_bits(p, r, src_start_bit=start_bit % 32, dst_start_bit=dst_start_bit, num_bits=nbits)
                else:
                    # in two uint32
                    r = self.declare_var("r", uint32, init=cast(src_uint8, ~uint32)[start_bit // 32])
                    s = self.declare_var("s", uint32, init=cast(src_uint8, ~uint32)[start_bit // 32 + 1])
                    r_bits = 32 - start_bit % 32
                    self.update_bits(p, r, src_start_bit=start_bit % 32, dst_start_bit=dst_start_bit, num_bits=r_bits)
                    self.update_bits(p, s, src_start_bit=0, dst_start_bit=j * 16 + r_bits, num_bits=nbits - r_bits)
            else:
                # use uint8 to extract
                if start_bit // 8 == end_bit // 8:
                    # in a single uint8
                    r = self.declare_var("r", uint32, init=src_uint8[start_bit // 8])
                    self.update_bits(p, r, src_start_bit=start_bit % 8, dst_start_bit=dst_start_bit, num_bits=nbits)
                else:
                    # in two uint8
                    r = self.declare_var("r", uint32, init=src_uint8[start_bit // 8])
                    s = self.declare_var("s", uint32, init=src_uint8[start_bit // 8 + 1])
                    r_bits = 8 - start_bit % 8
                    self.update_bits(p, r, src_start_bit=start_bit % 8, dst_start_bit=dst_start_bit, num_bits=r_bits)
                    self.update_bits(p, s, src_start_bit=0, dst_start_bit=j * 16 + r_bits, num_bits=nbits - r_bits)
        return p

    def cast_ix_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 2 != 0:
            self.cast_generic(src, dst)
            return

    def cast_ux_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 2 != 0:
            self.cast_generic(src, dst)
            return

        src_dtype = get_base_type(src.type)
        dst_dtype = get_base_type(dst.type)

        assert src_dtype.is_integer() and src_dtype.nbits <= 8 and int(src_dtype.min_value) >= 0
        assert dst_dtype == float16

        nbits = src_dtype.nbits
        src_uint8 = self.declare(
            v=tensor_pointer_var("src_uint8", shape=[cdiv(self.size * nbits, 8)], dtype="uint8"), init=cast(src, ~uint8)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        for i in range(self.size // 2):
            p = self.extract_bits_to_uint32(src_uint8=src_uint8, i=i, size=self.size, nbits=nbits, pos="low")
            h = self.declare_var("h", uint32, init=p | uint32(0x64006400))
            self.append(sub_f16x2(d=~h, a=h, b=uint32(0x64006400)))
            self.buffer_store(buf=dst_uint32, indices=[i], value=h)

    def cast_u4_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 8 != 0:
            self.cast_generic(src, dst)
            return

        src_uint32 = self.declare(
            v=tensor_pointer_var("src_uint32", shape=[self.size // 8], dtype="uint32"), init=cast(src, ~uint32)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        with self.for_range(extent=self.size // 8) as i:
            # cast 8 uint4b to float16 at each time
            # range for uint4b: 0 ~ 15

            # transform uint4b[7, 6, 5, 4, 3, 2, 1, 0] to float16 [4, 0], [5, 1], [6, 2], [7, 3]
            imm_lut = (0xF0 & 0xCC) | 0xAA
            a = self.declare(v=Var("a", type=uint32), init=src_uint32[i])
            b = self.declare(v=Var("b", type=uint32), init=src_uint32[i] >> 8)
            h = self.declare(v=tensor_var("h", shape=[4], dtype=uint32))
            self.append(lop3(~h[0], a, uint32(0x000F000F), uint32(0x64006400), imm_lut=imm_lut))  # fp16[4, 0] + 1024
            self.append(
                lop3(~h[1], a, uint32(0x00F000F0), uint32(0x64006400), imm_lut=imm_lut)
            )  # fp16[5, 1] * 16 + 1024
            self.append(lop3(~h[2], b, uint32(0x000F000F), uint32(0x64006400), imm_lut=imm_lut))  # fp16[6, 2] + 1024
            self.append(
                lop3(~h[3], b, uint32(0x00F000F0), uint32(0x64006400), imm_lut=imm_lut)
            )  # fp16[7, 3] * 16 + 1024
            self.append(sub_f16x2(d=~h[0], a=h[0], b=uint32(0x64006400)))  # x-1024 => fp16[4, 0]
            self.append(sub_f16x2(d=~h[2], a=h[2], b=uint32(0x64006400)))  # x-1024 => fp16[6, 2]
            self.append(
                fma_f16x2(d=~h[1], a=h[1], b=uint32(0x2C002C00), c=uint32(0xD400D400))
            )  # (x / 16) - 64 => fp16[5, 1]
            self.append(
                fma_f16x2(d=~h[3], a=h[3], b=uint32(0x2C002C00), c=uint32(0xD400D400))
            )  # (x / 16) - 64 => fp16[7, 3]
            self.append(prmt(d=~dst_uint32[i << 2 | 0], a=h[0], b=h[1], c=uint32(0x5410)))  # fp16[1, 0]
            self.append(prmt(d=~dst_uint32[i << 2 | 1], a=h[2], b=h[3], c=uint32(0x5410)))  # fp16[3, 2]
            self.append(prmt(d=~dst_uint32[i << 2 | 2], a=h[0], b=h[1], c=uint32(0x7632)))  # fp16[5, 4]
            self.append(prmt(d=~dst_uint32[i << 2 | 3], a=h[2], b=h[3], c=uint32(0x7632)))  # fp16[7, 6]

    def cast_f8e4m3_to_f16(self, src: Var, dst: Var) -> None:
        if self.size % 4 != 0:
            self.cast_generic(src, dst)
            return

        src_uint32 = self.declare(
            v=tensor_pointer_var("src_uint32", shape=[self.size // 4], dtype="uint32"), init=cast(src, ~uint32)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        # float8_e4m3: 1, 4, 3
        #     float16: 1, 5, 10
        with self.for_range(extent=self.size // 4) as i:
            p = self.declare(var("a", uint32))
            q = self.declare(var("b", uint32))
            self.append(prmt(d=~p, a=src_uint32[i], b=uint32(0), c=uint32(0x1404)))
            self.append(prmt(d=~q, a=src_uint32[i], b=uint32(0), c=uint32(0x3424)))
            # 0x3F80: 0b0011111110000000
            imm_lut = (0xF0 & 0xCC) | 0xAA
            self.append(lop3(d=~p, a=p >> 1, b=uint32(0x3F803F80), c=p & uint32(0x80008000), imm_lut=imm_lut))
            self.append(lop3(d=~q, a=q >> 1, b=uint32(0x3F803F80), c=q & uint32(0x80008000), imm_lut=imm_lut))
            # 0x5C00: 2^8 of float16
            self.append(mul_f16x2(d=~p, a=p, b=uint32(0x5C005C00)))
            self.append(mul_f16x2(d=~q, a=q, b=uint32(0x5C005C00)))
            self.buffer_store(buf=dst_uint32, indices=[i << 1], value=p)
            self.buffer_store(buf=dst_uint32, indices=[i << 1 | 1], value=q)

    def cast_f8e4m3_to_bf16(self, src: Var, dst: Var) -> None:
        if self.size % 4 != 0:
            self.cast_generic(src, dst)
            return

        src_uint32 = self.declare(
            v=tensor_pointer_var("src_uint32", shape=[self.size // 4], dtype="uint32"), init=cast(src, ~uint32)
        )
        dst_uint32 = self.declare(
            v=tensor_pointer_var("dst_uint32", shape=[self.size // 2], dtype="uint32"), init=cast(dst, ~uint32)
        )
        # float8_e4m3: 1, 4, 3
        #    bfloat16: 1, 8, 7
        with self.for_range(extent=self.size // 4) as i:
            p = self.declare(var("a", uint32))
            q = self.declare(var("b", uint32))
            self.append(prmt(d=~p, a=src_uint32[i], b=uint32(0), c=uint32(0x1404)))
            self.append(prmt(d=~q, a=src_uint32[i], b=uint32(0), c=uint32(0x3424)))
            # 0x07F0: 0b0000011111110000
            imm_lut = (0xF0 & 0xCC) | 0xAA
            self.append(lop3(d=~p, a=p >> 4, b=uint32(0x07F007F0), c=p & uint32(0x80008000), imm_lut=imm_lut))
            self.append(lop3(d=~q, a=q >> 4, b=uint32(0x07F007F0), c=q & uint32(0x80008000), imm_lut=imm_lut))
            # 0x7B80: 2^120 of bfloat16
            self.append(mul_bf16x2(d=~p, a=p, b=uint32(0x7B807B80)))
            self.append(mul_bf16x2(d=~q, a=q, b=uint32(0x7B807B80)))
            self.buffer_store(buf=dst_uint32, indices=[i << 1], value=p)
            self.buffer_store(buf=dst_uint32, indices=[i << 1 | 1], value=q)

    def update_bits(
        self, dst_uint32: Expr, src_uint32: Expr, src_start_bit: int, dst_start_bit: int, num_bits: int
    ) -> None:
        # the dst_uint32 must be zero on the specified bits
        bit_offset = src_start_bit - dst_start_bit
        assert 0 <= src_start_bit <= 32 - num_bits and 0 <= dst_start_bit <= 32 - num_bits
        if bit_offset >= 0:
            src_uint32 = src_uint32 >> bit_offset
        else:
            src_uint32 = src_uint32 << -bit_offset
        mask = ((uint32(1) << num_bits) - 1) << dst_start_bit
        self.append(lop3(d=~dst_uint32, a=dst_uint32, b=src_uint32, c=mask, imm_lut=lambda a, b, c: a | (b & c)))

    def cast_f6e3m2_to_bf16(self, src: Var, dst: Var) -> None:
        if self.size % 2 != 0:
            self.cast_generic(src, dst)
            return

        # f6e3m2: 1, 3, 2
        #   bf16: 1, 8, 7
        src_uint8 = self.declare(tensor_pointer_var("src_uint8", [self.size * 6 // 8], uint8), init=cast(src, ~uint8))
        dst_uint32 = self.declare(tensor_pointer_var("dst_uint32", [self.size // 2], uint32), init=cast(dst, ~uint32))

        for i in range(self.size // 2):
            p = self.extract_bits_to_uint32(src_uint8=src_uint8, i=i, size=self.size, nbits=6, pos="high")

            # adjust bits position where 0x03F0: 0b0000001111110000
            self.append(
                lop3(
                    d=~p, a=p >> 5, b=uint32(0x03F003F0), c=p & uint32(0x80008000), imm_lut=lambda a, b, c: (a & b) | c
                )
            )

            # adjust to fix e_bias
            # 2^124 of bfloat16: 0x7D80
            self.append(mul_bf16x2(d=~p, a=p, b=uint32(0x7D807D80)))

            self.buffer_store(dst_uint32, indices=[i], value=p)

    def cast_from_subbyte_16bit_float(self, src: Var, dst: Var) -> None:
        if self.size % 2 != 0:
            self.cast_generic(src, dst)
            return

        src_dtype = get_base_type(src.type)
        dst_dtype = get_base_type(dst.type)

        assert src_dtype.is_float() and dst_dtype.nbits < 8
        assert dst_dtype.is_float() and dst_dtype.nbits == 16

        nbits: int = src_dtype.nbits
        src_uint8 = self.declare(
            tensor_pointer_var("src_uint8", [(self.size * src_dtype.nbits + 7) // 8], uint8), init=cast(src, ~uint8)
        )
        dst_uint32 = self.declare(tensor_pointer_var("dst_uint32", [self.size // 2], uint32), init=cast(dst, ~uint32))

        for i in range(self.size // 2):
            p = self.extract_bits_to_uint32(src_uint8=src_uint8, i=i, size=self.size, nbits=nbits, pos="high")

            assert isinstance(src_dtype, FloatSubbyteType)
            assert isinstance(dst_dtype, FloatType)

            # adjust bits position to align the intersection of mantissa and exponent for the two data types
            uint16_mask = ((1 << (src_dtype.nbits - 1)) - 1) << (dst_dtype.mantissa_nbits - src_dtype.mantissa_nbits)
            uint32_mask = uint32(uint16_mask << 16 | uint16_mask)
            self.append(
                lop3(
                    d=~p,
                    a=p >> (dst_dtype.exponent_nbits - src_dtype.exponent_nbits),
                    b=uint32_mask,
                    c=p & uint32(0x80008000),
                    imm_lut=lambda a, b, c: (a & b) | c,
                )
            )

            # adjust to fix e_bias
            e_bias_adjust = 2 ** (dst_dtype.exponent_nbits - 1) - 2 ** (src_dtype.exponent_nbits - 1)
            power_e_bias_adjust_uint16 = (
                e_bias_adjust + 2 ** (dst_dtype.exponent_nbits - 1) - 1
            ) << dst_dtype.mantissa_nbits
            power_e_bias_adjust_uint32 = uint32(power_e_bias_adjust_uint16 << 16 | power_e_bias_adjust_uint16)

            if dst_dtype == bfloat16:
                mul_func = mul_bf16x2
            elif dst_dtype == float16:
                mul_func = mul_f16x2
            else:
                raise NotImplementedError()
            self.append(mul_func(d=~p, a=p, b=power_e_bias_adjust_uint32))

            self.buffer_store(dst_uint32, indices=[i], value=p)


@register_emitter(CastInst, target=amdgpu_any)
class AmdgpuCastInstEmitter(CastInstBaseEmitter):
    def __init__(self, codegen):
        super().__init__(codegen)
        self.specialized_cast.update({})
