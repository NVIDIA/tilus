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
import tilus
from tilus.testing.requires import requires


class DemoBlockCluster(tilus.Script):
    def __init__(self, cluster_blocks):
        super().__init__()
        self.cluster_blocks = cluster_blocks

    def __call__(self):
        self.attrs.blocks = [2, 2, 2]
        self.attrs.cluster_blocks = self.cluster_blocks
        self.attrs.warps = 4

        self.printf("blockIdx: [%d, %d, %d]\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z)


@requires.nvgpu_sm90
def test_script_blocks_per_cluster_post_sm90():
    kernel = DemoBlockCluster((2, 2, 1))
    kernel()


@requires.nvgpu_sm80
def test_script_blocks_per_cluster_pre_sm90():
    kernel = DemoBlockCluster((1, 1, 1))
    kernel()
