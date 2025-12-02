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
from tilus.ir.layout.ops import shared_column_major, shared_layout, shared_reshape, shared_row_major


@pytest.mark.parametrize(
    "layout, new_shape, expected",
    [
        (shared_row_major(12), [3, 4], shared_row_major(3, 4)),
        (shared_layout([12], [3, 4], [1, 3]), [3, 4], shared_column_major(3, 4)),
        (shared_layout([12], [3, 4], [1, 3]), [4, 3], None),
    ],
)
def test_shared_layout_reshape(layout, new_shape, expected):
    if expected is None:
        # expect a runtime error
        with pytest.raises(RuntimeError):
            shared_reshape(layout, new_shape)
    else:
        actual = shared_reshape(layout, new_shape)
        assert actual == expected


if __name__ == "__main__":
    pytest.main([__file__])
