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

from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import Tcgen05LoadStoreShapeKind
from tilus.ir.layout.ops.register_ops import local, register_layout, spatial
from tilus.ir.layout.register_layout import RegisterLayout, visualize_layout


@functools.cache
def get_ldst_layout(shape: Tcgen05LoadStoreShapeKind) -> RegisterLayout:
    # see https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-layout
    if shape == Tcgen05LoadStoreShapeKind.R32x32B:
        return spatial(32, 1)
    elif shape == Tcgen05LoadStoreShapeKind.R16x64B:
        """
        ┌───────┬───────┐
        │ 0: 0  │ 2: 0  │
        ├───────┼───────┤
        │ 4: 0  │ 6: 0  │
        ├───────┼───────┤
        │ 8: 0  │ 10: 0 │
        ├───────┼───────┤
        │ 12: 0 │ 14: 0 │
        ├───────┼───────┤
        │ 16: 0 │ 18: 0 │
        ├───────┼───────┤
        │ 20: 0 │ 22: 0 │
        ├───────┼───────┤
        │ 24: 0 │ 26: 0 │
        ├───────┼───────┤
        │ 28: 0 │ 30: 0 │
        ├───────┼───────┤
        │ 1: 0  │ 3: 0  │
        ├───────┼───────┤
        │ 5: 0  │ 7: 0  │
        ├───────┼───────┤
        │ 9: 0  │ 11: 0 │
        ├───────┼───────┤
        │ 13: 0 │ 15: 0 │
        ├───────┼───────┤
        │ 17: 0 │ 19: 0 │
        ├───────┼───────┤
        │ 21: 0 │ 23: 0 │
        ├───────┼───────┤
        │ 25: 0 │ 27: 0 │
        ├───────┼───────┤
        │ 29: 0 │ 31: 0 │
        └───────┴───────┘
        """
        return register_layout(shape=[16, 2], mode_shape=[2, 8, 2], spatial_modes=[1, 2, 0], local_modes=[])
    elif shape == Tcgen05LoadStoreShapeKind.R16x128B:
        return local(2, 1).spatial(8, 4)
    elif shape == Tcgen05LoadStoreShapeKind.R16x256B:
        return local(2, 1).spatial(8, 4).local(1, 2)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


if __name__ == "__main__":
    layout_0 = spatial(32, 1)
    layout_1 = register_layout(shape=[16, 2], mode_shape=[2, 8, 2], spatial_modes=[1, 2, 0], local_modes=[])
    layout_3 = get_ldst_layout(Tcgen05LoadStoreShapeKind.R16x64B)
    print(layout_3)
    print(visualize_layout(layout_0))
    print(visualize_layout(layout_1))
    print(visualize_layout(layout_3))
