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
import tilus
import torch
from tilus import boolean, int32, uint8
from tilus.utils import cdiv


class ClusterLaunchControlExample(tilus.Script):
    def __init__(self, cluster_blocks: int, warps: int, num_stages: int):
        super().__init__()
        self.cluster_blocks = cluster_blocks
        self.warps = warps
        self.num_stages = num_stages
        self.block_n = warps * 32

    def __call__(self, n: int32, p_out: ~int32) -> None:
        """
        workers:
        - tile scheduler (warp 0, cluster id 0)
        - main workers (all warps in the block cluster).

        pipelines:
        - cluster launch control pipeline
        """
        self.attrs.cluster_blocks = self.cluster_blocks
        self.attrs.warps = self.warps
        self.attrs.blocks = cdiv(n, 32 * self.warps)

        g_out = self.global_view(p_out, dtype=int32, shape=[n])

        self.shared_tensor(
            dtype=uint8, shape=[156 * 1024]
        )  # 156KB shared memory used to make each SM only have one block

        is_first_block_in_cluster: boolean = self.cluster.block_rank() == 0

        cancel_response = self.shared_tensor(dtype=int32, shape=[self.num_stages, 4])
        producer_mbarriers = self.mbarrier.alloc(count=[self.warps * 32 for _ in range(self.num_stages)])
        consumer_mbarriers = self.mbarrier.alloc(count=[1 for _ in range(self.num_stages)])
        producer_phase: int32 = self.mbarrier.producer_initial_phase
        consumer_phase: int32 = self.mbarrier.consumer_initial_phase
        producer_stage: int32 = 0
        consumer_stage: int32 = 0

        # wait all barriers are allocated and initialized in all blocks in the cluster
        # it's necessary. If not, some blocks may run and modify the barrier of other blocks before they are initialized.
        self.cluster.sync()

        offset_n: int32 = self.blockIdx.x * self.block_n

        # persistent loop
        while True:
            # tile scheduler, as producer of clc pipeline
            if is_first_block_in_cluster:
                with self.thread_group(thread_begin=0, num_threads=32):
                    self.mbarrier.wait(producer_mbarriers[producer_stage], phase=producer_phase)
                    self.clc.try_cancel(
                        cancel_response[producer_stage], mbarrier=consumer_mbarriers[producer_stage], multicast=True
                    )
                    producer_stage = (1 + producer_stage) % self.num_stages
                    producer_phase = producer_phase ^ (producer_stage == 0)

            # store the result
            self.store_global(
                dst=g_out, src=self.register_tensor(dtype=int32, shape=[self.block_n], init=1), offsets=[offset_n]
            )

            # consumer of clc pipeline
            self.mbarrier.wait(consumer_mbarriers[consumer_stage], phase=consumer_phase)
            is_valid, blockIdx = self.clc.query_response(cancel_response[consumer_stage])
            self.mbarrier.arrive(producer_mbarriers[consumer_stage])
            consumer_stage = (1 + consumer_stage) % self.num_stages
            consumer_phase = consumer_phase ^ (consumer_stage == 0)
            if is_valid:
                offset_n = (blockIdx.x + self.cluster.block_id().x) * self.block_n
            else:
                break


@tilus.testing.requires.nvgpu_sm100
@pytest.mark.parametrize("cluster_blocks", [2, 4])
@pytest.mark.parametrize("num_stages", [2, 3, 4])
@pytest.mark.parametrize("warps", [4, 8])
def test_cluster_launch_control(cluster_blocks, num_stages, warps):
    segments = 200
    n = 128 * segments

    kernel = ClusterLaunchControlExample(
        cluster_blocks=cluster_blocks,
        warps=warps,
        num_stages=num_stages,
    )

    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    kernel(n, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.ones_like(out))


if __name__ == "__main__":
    pytest.main([__file__])
