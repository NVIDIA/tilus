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
from tilus.ir.layout.register_layout import canonicalize_layout, register_layout


@pytest.mark.parametrize(
    "original, expected",
    [
        [
            register_layout(shape=[6], mode_shape=[2, 3], spatial_modes=[0, 1], local_modes=[]),
            register_layout(shape=[6], mode_shape=[6], spatial_modes=[0], local_modes=[]),
        ],
        [
            register_layout(shape=[6], mode_shape=[2, 3], spatial_modes=[1, 0], local_modes=[]),
            register_layout(shape=[6], mode_shape=[2, 3], spatial_modes=[1, 0], local_modes=[]),
        ],
        [
            register_layout(shape=[6], mode_shape=[2, 3], spatial_modes=[1, 0], local_modes=[]),
            register_layout(shape=[6], mode_shape=[2, 3], spatial_modes=[1, 0], local_modes=[]),
        ],
        [
            register_layout(shape=[4, 6, 5], mode_shape=[2, 2, 2, 3, 5], spatial_modes=[0, 1], local_modes=[2, 3, 4]),
            register_layout(shape=[4, 6, 5], mode_shape=[4, 6, 5], spatial_modes=[0], local_modes=[1, 2]),
        ],
        [
            register_layout(
                shape=[4, 5, 6, 5], mode_shape=[2, 2, 5, 2, 3, 5], spatial_modes=[2, 0, 1], local_modes=[3, 4, 5]
            ),
            register_layout(shape=[4, 5, 6, 5], mode_shape=[4, 5, 6, 5], spatial_modes=[1, 0], local_modes=[2, 3]),
        ],
    ],
)
def test_canonicalize(original, expected):
    assert canonicalize_layout(original) == expected, f"Canonicalization failed: {original} -> {expected}"
