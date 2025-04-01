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
from hidet.ir.dtypes.integer import uint32
from hidet.ir.dtypes.integer_subbyte import (
    IntegerSubbyteType,
)

i7 = int7b = IntegerSubbyteType("int7b", "i7", uint32, 7, True, -64, 63)
i6 = int6b = IntegerSubbyteType("int6b", "i6", uint32, 6, True, -32, 31)
i5 = int5b = IntegerSubbyteType("int5b", "i5", uint32, 5, True, -16, 15)

u7 = uint7b = IntegerSubbyteType("uint7b", "u7", uint32, 7, False, 0, 127)
u6 = uint6b = IntegerSubbyteType("uint6b", "u6", uint32, 6, False, 0, 63)
u5 = uint5b = IntegerSubbyteType("uint5b", "u5", uint32, 5, False, 0, 31)
