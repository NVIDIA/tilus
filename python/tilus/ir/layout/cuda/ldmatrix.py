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
from __future__ import annotations

import functools
from dataclasses import dataclass

from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.ops import column_spatial, spatial


@dataclass(frozen=True, eq=False)
class LoadMatrixConfig:
    nbytes: int
    trans: bool
    ldmatrix_layout: RegisterLayout

    @staticmethod
    @functools.cache
    def all() -> tuple[LoadMatrixConfig, ...]:
        return (
            LoadMatrixConfig(1, False, spatial(8, 4).local(1, 4)),
            LoadMatrixConfig(2, False, spatial(8, 4).local(1, 2)),
            LoadMatrixConfig(4, False, spatial(8, 4)),
            LoadMatrixConfig(2, True, column_spatial(4, 8).local(2, 1)),
        )


@dataclass(frozen=True, eq=False)
class StoreMatrixConfig:
    nbytes: int
    trans: bool
    stmatrix_layout: RegisterLayout

    @staticmethod
    @functools.cache
    def all() -> tuple[StoreMatrixConfig, ...]:
        return (
            StoreMatrixConfig(1, False, spatial(8, 4).local(1, 4)),
            StoreMatrixConfig(2, False, spatial(8, 4).local(1, 2)),
            StoreMatrixConfig(4, False, spatial(8, 4)),
            StoreMatrixConfig(2, True, column_spatial(4, 8).local(2, 1)),
        )
