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
from tilus.ir.layout.ops import register_layout, reshape, spatial


@pytest.mark.parametrize(
    "original, shape, expect",
    [
        [spatial(2, 4), [2, 2, 2], spatial(2, 2, 2)],
        [
            spatial(2, 4).local(3, 2).spatial(6, 8),
            [2, 3, 6, 4, 2, 8],
            register_layout(
                shape=[2, 3, 6, 4, 2, 8], mode_shape=[2, 3, 6, 4, 2, 8], spatial_modes=[0, 3, 2, 5], local_modes=[1, 4]
            ),
        ],
        [
            register_layout(
                shape=[2, 3, 6, 4, 2, 8], mode_shape=[2, 3, 6, 4, 2, 8], spatial_modes=[0, 3, 2, 5], local_modes=[1, 4]
            ),
            [36, 64],
            register_layout(
                shape=[36, 64], mode_shape=[2, 3, 6, 4, 2, 8], spatial_modes=[0, 3, 2, 5], local_modes=[1, 4]
            ),
        ],
    ],
)
def test_reshape(original, shape, expect):
    actual = reshape(original, shape)
    assert actual == expect, f"Reshape failed: {original} -> {shape}, expect {expect}, got {actual}"
