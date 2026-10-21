# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import pytest
from tilus.ir.layout import global_layout as global_layout_module
from tilus.ir.layout.ops import global_ops as global_ops_module
from tilus.ir.utils.veceval import vectorized_evaluate


def evaluate_offsets(layout) -> np.ndarray:
    shape = [int(s) for s in layout.shape]
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return vectorized_evaluate(layout.offset, {axis: grid[i] for i, axis in enumerate(layout.axes)})


@pytest.mark.parametrize("module", [global_layout_module, global_ops_module])
def test_global_compose_row_major(module):
    # a 2x2 grid of 2x3 row-major tiles, where the tiles are also stored in row-major order
    layout = module.global_compose(module.global_row_major(2, 2), module.global_row_major(2, 3))
    expected = np.array(
        [
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5, 9, 10, 11],
            [12, 13, 14, 18, 19, 20],
            [15, 16, 17, 21, 22, 23],
        ]
    )
    assert np.array_equal(evaluate_offsets(layout), expected)


@pytest.mark.parametrize("module", [global_layout_module, global_ops_module])
def test_global_compose_column_major_tiles(module):
    # a 2x2 grid of 2x3 column-major tiles, where the tiles are stored in row-major order
    layout = module.global_compose(module.global_row_major(2, 2), module.global_column_major(2, 3))
    expected = np.array(
        [
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
            [12, 14, 16, 18, 20, 22],
            [13, 15, 17, 19, 21, 23],
        ]
    )
    assert np.array_equal(evaluate_offsets(layout), expected)


if __name__ == "__main__":
    pytest.main([__file__])
