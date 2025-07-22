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
from tilus.ir.layout.register_layout_ops import local, register_layout, spatial


def test_spatial():
    for actual, expected in [
        [spatial(), register_layout(shape=[], mode_shape=[], spatial_modes=[], local_modes=[])],  # identity
        [spatial(3, 4), register_layout(shape=[3, 4], mode_shape=[3, 4], spatial_modes=[0, 1], local_modes=[])],
        [
            spatial(1, 1, 2, 1, 3, 1),
            register_layout(shape=[1, 1, 2, 1, 3, 1], mode_shape=[2, 3], spatial_modes=[0, 1], local_modes=[]),
        ],
    ]:
        assert actual == expected


def test_local():
    for actual, expected in [
        [local(), register_layout(shape=[], mode_shape=[], spatial_modes=[], local_modes=[])],  # identity
        [local(3, 4), register_layout(shape=[3, 4], mode_shape=[3, 4], spatial_modes=[], local_modes=[0, 1])],
        [
            local(1, 1, 2, 1, 3, 1),
            register_layout(shape=[1, 1, 2, 1, 3, 1], mode_shape=[2, 3], spatial_modes=[], local_modes=[0, 1]),
        ],
    ]:
        assert actual == expected
