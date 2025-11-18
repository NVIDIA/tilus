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
from tilus import uint32


class PingPongExample(tilus.Script):
    def __call__(self):
        self.attrs.blocks = 1
        self.attrs.warps = 8

        a_ready, b_ready = self.mbarrier.alloc(count=[128, 128]).tolist()
        a_phase: uint32 = 0
        b_phase: uint32 = 1
        num_rounds = 10

        with self.thread_group(thread_begin=0, num_threads=128):
            for round in self.range(num_rounds):
                self.printf("[A][round=%d] waiting, phase=%d\n", round, a_phase)
                self.mbarrier.wait(a_ready, phase=a_phase)
                self.printf("[A][round=%d] proceeding\n", round)
                a_phase = a_phase ^ 1
                self.printf("[A][round=%d] finished\n", round)
                self.mbarrier.arrive(b_ready)

        with self.thread_group(thread_begin=128, num_threads=128):
            for round in self.range(num_rounds):
                self.printf("[B][round=%d] waiting, phase=%d\n", round, b_phase)
                self.mbarrier.wait(b_ready, phase=b_phase)
                self.printf("[B][round=%d] proceeding\n", round)
                b_phase = b_phase ^ 1
                self.printf("[B][round=%d] finished\n", round)
                self.mbarrier.arrive(a_ready)


@tilus.testing.requires.nvgpu_sm90
def test_ping_pong_example():
    kernel = PingPongExample()
    kernel()
    torch.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
