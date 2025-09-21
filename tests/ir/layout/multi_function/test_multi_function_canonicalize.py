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
from tilus.ir.layout.mfunction import canonicalize_multi_function, multi_function


@pytest.mark.parametrize(
    "a, expected",
    [
        (
            multi_function([2, 3], [2, 1, 3], [0, 2]),
            multi_function([2, 3], [2, 3], [0, 1]),
        ),
        (
            multi_function([24, 5, 6], [2, 3, 4, 5, 6], [0, 1, 2, 4]),
            multi_function([24, 5, 6], [24, 5, 6], [0, 2]),
        ),
    ],
)
def test_multi_function_canonicalization(a, expected):
    actual = canonicalize_multi_function(a)
    assert actual == expected, f"Expected {expected}, but got {actual} for {a}"
