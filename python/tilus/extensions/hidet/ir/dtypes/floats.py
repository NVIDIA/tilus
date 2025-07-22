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
from hidet.ir.dtypes.floats import FloatType, bfloat16, float16, float32, float64, tfloat32

f8e4m3 = float8_e4m3 = FloatType(
    "float8_e4m3", "f8e4m3", 1, min_value=float(-448), max_value=float(448), eps=2 ** (-2), smallest_normal=2 ** (-6)
)
f8e5m2 = float8_e5m2 = FloatType(
    "float8_e5m2",
    "f8e5m2",
    1,
    min_value=float(-57344),
    max_value=float(57344),
    eps=2 ** (-2),
    smallest_normal=2 ** (-14),
)

# todo: add the mantissa and exponent bits
_mantissa_bits = {
    "float64": 52,
    "float32": 23,
    "tfloat32": 10,
    "float16": 10,
    "bfloat16": 7,
    "float8_e5m2": 2,
    "float8_e4m3": 3,
}
_exponent_bits = {
    "float64": 11,
    "float32": 8,
    "tfloat32": 8,
    "float16": 5,
    "bfloat16": 8,
    "float8_e5m2": 5,
    "float8_e4m3": 4,
}

for float_dtype in [float64, float32, tfloat32, bfloat16, float16, float8_e5m2, float8_e4m3]:
    float_dtype.mantissa_nbits = _mantissa_bits[float_dtype.name]  # type: ignore
    float_dtype.exponent_nbits = _exponent_bits[float_dtype.name]  # type: ignore
