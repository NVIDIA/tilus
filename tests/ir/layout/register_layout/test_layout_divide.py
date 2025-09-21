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
from tilus.ir.layout.ops import divide, local, spatial


@pytest.mark.parametrize(
    "lhs, rhs, expect",
    [
        [spatial(2, 4), spatial(1, 2), spatial(2, 2)],
        [spatial(2, 4).local(2, 1), local(2, 1), spatial(2, 4)],
        [spatial(2, 4).local(2, 1), spatial(1, 2).local(2, 1), spatial(2, 2)],
        [spatial(32, 1).local(1, 32), spatial(32, 1), local(1, 32)],
        # split first element in suffix (mode refinement)
        [spatial(6), spatial(3), spatial(2)],
        # refinement without locals
        [spatial(2, 12), spatial(1, 3), spatial(2, 4)],
        # remove locals entirely
        [spatial(2, 12).local(2, 3), local(2, 3), spatial(2, 12)],
    ],
)
def test_divide(lhs, rhs, expect):
    actual = divide(lhs, rhs)
    assert actual == expect, f"Divide failed: {lhs} / {rhs}, expect {expect}, got {actual}"
