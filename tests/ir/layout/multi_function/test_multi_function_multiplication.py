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
import pytest
from tilus.ir.layout.mfunction import MultiFunction, multi_function


@pytest.mark.parametrize(
    "a, b, expected",
    [
        [
            multi_function([2, 3, 4], [2, 3, 4], [0, 2]),
            multi_function([8], [2, 2, 2], [0, -5, 2]),
            multi_function([2, 3, 4], [2, 3, 2, 2], [0, -5, 3]),
        ],
        [
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, -2]),
            multi_function([16], [2, 2, 2, 2], [0, -5, 2]),
            multi_function([2, 3, 4], [2, 3, 2, 2], [0, -5, 3]),
        ],
        [
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, -2]),
            multi_function([16], [2, 2, 2, 2], [0, -5, 2, 3]),
            multi_function([2, 3, 4], [2, 3, 2, 2], [0, -5, 3, -2]),
        ],
    ],
)
def test_multi_function_multiplication(
    a: MultiFunction,
    b: MultiFunction,
    expected: MultiFunction,
):
    actual = a * b
    assert actual == expected, f"Expected {expected}, but got {actual} for {a} * {b}"
