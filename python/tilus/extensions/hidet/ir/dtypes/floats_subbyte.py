# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

from hidet.ir.dtypes.floats import FloatInfo, FloatType


class FloatSubbyteType(FloatType):
    def __init__(self, name: str, short_name: str, nbits: int, exponent_nbits: int, mantissa_nbits: int) -> None:
        self._nbits: int = nbits
        self._exponent_nbits: int = exponent_nbits
        self._mantissa_nbits: int = mantissa_nbits
        self._exponent_bias: int = (1 << (exponent_nbits - 1)) - 1
        finfo = self._calculate_finfo()
        super().__init__(
            name=name,
            short_name=short_name,
            nbytes=-1,
            min_value=finfo.min,
            max_value=finfo.max,
            eps=finfo.eps,
            smallest_normal=finfo.smallest_normal,
        )

    def _calculate_finfo(self) -> FloatInfo:
        # we do not include 'nan' or 'inf' in the sub-byte floating type representation
        # to fully use its exponent bits
        e_bits = self._exponent_nbits
        m_bits = self._mantissa_nbits
        e_bias = self._exponent_bias
        return FloatInfo(
            bits=e_bits + m_bits + 1,
            eps=math.pow(2.0, -m_bits),
            min=-math.pow(2.0, (1 << e_bits) - 1 - e_bias) * (2.0 - math.pow(2.0, -m_bits)),
            max=math.pow(2.0, (1 << e_bits) - 1 - e_bias) * (2.0 - math.pow(2.0, -m_bits)),
            smallest_normal=math.pow(2.0, 1 - e_bias),
            dtype=self,
        )

    @property
    def nbytes(self):
        raise TypeError(f"Cannot access nbytes property for the type({self}")

    @property
    def nbits(self):
        return self._nbits

    @property
    def exponent_nbits(self):
        return self._exponent_nbits

    @property
    def mantissa_nbits(self):
        return self._mantissa_nbits

    def is_float_subbyte(self) -> bool:
        return True

    def is_subbyte(self):
        return True

    def finfo(self) -> FloatInfo:
        return FloatInfo(
            bits=self.nbits,
            eps=self._eps,
            max=self._max_value,
            min=self._min_value,
            smallest_normal=self._smallest_normal,
            dtype=self,
        )


# float7
f7e5m1 = float7_e5m1 = FloatSubbyteType("float7_e5m1", "f7e5m1", 7, 5, 1)
f7e4m2 = float7_e4m2 = FloatSubbyteType("float7_e4m2", "f7e4m2", 7, 4, 2)
f7e3m3 = float7_e3m3 = FloatSubbyteType("float7_e3m3", "f7e3m3", 7, 3, 3)
f7e2m4 = float7_e2m4 = FloatSubbyteType("float7_e2m4", "f7e2m4", 7, 2, 4)

# float6
f6e4m1 = float6_e4m1 = FloatSubbyteType("float6_e4m1", "f6e4m1", 6, 4, 1)
f6e3m2 = float6_e3m2 = FloatSubbyteType("float6_e3m2", "f6e3m2", 6, 3, 2)
f6e2m3 = float6_e2m3 = FloatSubbyteType("float6_e2m3", "f6e2m3", 6, 2, 3)

# float5
f5e3m1 = float5_e3m1 = FloatSubbyteType("float5_e3m1", "f5e3m1", 5, 3, 1)
f5e2m2 = float5_e2m2 = FloatSubbyteType("float5_e2m2", "f5e2m2", 5, 2, 2)

# float4
f4e2m1 = float4_e2m1 = FloatSubbyteType("float4_e2m1", "f4e2m1", 4, 2, 1)

# float3
f3e1m1 = float3_e1m1 = FloatSubbyteType("float3_e1m1", "f3e1m1", 3, 1, 1)
