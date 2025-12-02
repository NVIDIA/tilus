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
from hidet.utils import prod
from hidet.ir.dtypes import int32
from hidet.ir.utils.index_transform import index_serialize, index_deserialize
from tilus.ir.layout.ops import shared_row_major, shared_reshape


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([16, 16], [32, 8]),
        ([8, 32], [16, 16]),
        ([4, 4, 4], [8, 2, 4]),
        ([2, 8, 4], [4, 4, 4]),
        ([2, 3, 4, 5], [24, 1, 5])
    ]
)
def test_shared_layout_reshape(src_shape, dst_shape):
    layout = shared_row_major(*src_shape)
    count = prod(src_shape)
    reshaped_layout = shared_reshape(layout, dst_shape)

    assert prod(src_shape) == prod(dst_shape)

    for i in range(count):
        src_indices = index_deserialize(int32(i), src_shape)
        dst_indices = index_deserialize(int32(i), dst_shape)
        src_offset = layout(*src_indices)
        dst_offset = reshaped_layout(*dst_indices)
        assert src_offset == dst_offset, f"src_shape: {src_shape}, dst_shape: {dst_shape}, src_indices: {src_indices}, dst_indices: {dst_indices}, src_offset: {src_offset}, dst_offset: {dst_offset}"

if __name__ == "__main__":
    pytest.main([__file__])
