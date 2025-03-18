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
from hidet.ir.type import DataType
from hidet.ir.dtypes.integer import IntegerType, IntInfo, uint8, uint32


class IntegerSubbyteType(IntegerType):
    def __init__(
        self, name: str, short_name: str, storage: DataType, nbits: int, signed: bool, min_value: int, max_value: int
    ) -> None:
        nbytes = storage.nbytes
        super().__init__(name, short_name, nbytes, min_value, max_value)
        self._storage: DataType = storage
        self._nbits: int = nbits
        self._bits_mask: int = (1 << self._nbits) - 1
        self._sign_mask: int = 1 << (self._nbits - 1) if self.signedness() else 0

    @property
    def storage(self):
        return self._storage

    @property
    def nbytes(self):
        raise TypeError(f"Cannot access nbytes property for the type({self}")

    @property
    def nbits(self):
        return self._nbits

    @property
    def bits_mask(self):
        return self._bits_mask

    @property
    def sign_mask(self):
        return self._sign_mask

    def is_subbyte(self) -> bool:
        return True

    def is_integer_subbyte(self):
        return True

    def is_float_subbyte(self) -> bool:
        return False

    def iinfo(self) -> IntInfo:
        return IntInfo(self._nbits, self._max_value, self._min_value, self)


i7 = int7b = IntegerSubbyteType("int7b", "i7", uint32, 7, True, -64, 63)
i6 = int6b = IntegerSubbyteType("int6b", "i6", uint32, 6, True, -32, 31)
i5 = int5b = IntegerSubbyteType("int5b", "i5", uint32, 5, True, -16, 15)
i4 = int4b = IntegerSubbyteType("int4b", "i4", uint8, 4, True, -8, 7)
i3 = int3b = IntegerSubbyteType("int3b", "i3", uint32, 3, True, -4, 3)
i2 = int2b = IntegerSubbyteType("int2b", "i2", uint8, 2, True, -2, 1)
i1 = int1b = IntegerSubbyteType("int1b", "i1", uint8, 1, True, -1, 0)

u7 = uint7b = IntegerSubbyteType("uint7b", "u7", uint32, 7, False, 0, 127)
u6 = uint6b = IntegerSubbyteType("uint6b", "u6", uint32, 6, False, 0, 63)
u5 = uint5b = IntegerSubbyteType("uint5b", "u5", uint32, 5, False, 0, 31)
u4 = uint4b = IntegerSubbyteType("uint4b", "u4", uint8, 4, False, 0, 15)
u3 = uint3b = IntegerSubbyteType("uint3b", "u3", uint32, 3, False, 0, 7)
u2 = uint2b = IntegerSubbyteType("uint2b", "u2", uint8, 2, False, 0, 3)
u1 = uint1b = IntegerSubbyteType("uint1b", "u1", uint8, 1, False, 0, 1)
