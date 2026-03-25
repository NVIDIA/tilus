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
import tilus.testing
import torch
from tilus import float16, int32
from tilus.utils import cdiv


class ThreadGroupExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block: int = 64

    def __call__(self, n: int32, x_ptr: ~float16, y_ptr: ~float16) -> None:
        self.attrs.blocks = cdiv(n, self.block)
        self.attrs.warps = 4

        g_x = self.global_view(x_ptr, dtype=float16, shape=[n])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[n])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block])
        offset = self.blockIdx.x * self.block

        with self.thread_group(0, num_threads=64):
            self.store_shared(s_x, self.load_global(g_x, offsets=[offset], shape=[self.block]))
            self.sync()
        self.sync()

        r_x = self.load_shared(s_x)
        r_x += 1
        self.store_shared(dst=s_x, src=r_x)
        self.sync()

        with self.thread_group(64, num_threads=64):
            self.store_global(dst=g_y, src=self.load_shared(s_x), offsets=[offset])

        self.sync()


@tilus.testing.requires.nvgpu_sm90
def test_thread_group():
    n = 1024
    x = torch.randn(n, dtype=torch.float16, device="cuda")
    y = torch.zeros(n, dtype=torch.float16, device="cuda")

    kernel = ThreadGroupExample()
    kernel(n, x, y)

    expect = x + 1
    actual = y
    torch.testing.assert_close(actual=actual, expected=expect)


class ElectAnyThreadGroupExample(tilus.Script):
    """Test elect-any (thread_begin=-1) semantics.

    Uses single_thread() (elect-any, num_threads=1) to have one elected thread
    write a value to shared memory, then all threads read it back. This verifies
    that exactly one thread executes the body and the result is visible to all.
    """

    def __init__(self):
        super().__init__()
        self.block: int = 64

    def __call__(self, n: int32, x_ptr: ~float16, y_ptr: ~float16) -> None:
        self.attrs.blocks = cdiv(n, self.block)
        self.attrs.warps = 4

        g_x = self.global_view(x_ptr, dtype=float16, shape=[n])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[n])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block])
        offset = self.blockIdx.x * self.block

        # Load global -> shared using all threads
        self.store_shared(s_x, self.load_global(g_x, offsets=[offset], shape=[self.block]))
        self.sync()

        # Use elect-any single_thread (thread_begin=-1) to increment all elements.
        # Only one elected thread per warp_group should execute this.
        # We use thread_group to narrow to a single warp first, then elect one thread.
        with self.single_warp(0):
            with self.single_thread():  # elect-any: uses elect.sync
                r_x = self.load_shared(s_x)
                r_x += 1
                self.store_shared(dst=s_x, src=r_x)
        self.sync()

        # Store shared -> global using all threads
        self.store_global(dst=g_y, src=self.load_shared(s_x), offsets=[offset])


@tilus.testing.requires.nvgpu_sm90
def test_elect_any_thread_group():
    n = 1024
    x = torch.randn(n, dtype=torch.float16, device="cuda")
    y = torch.zeros(n, dtype=torch.float16, device="cuda")

    kernel = ElectAnyThreadGroupExample()
    kernel(n, x, y)

    expect = x + 1
    actual = y
    torch.testing.assert_close(actual=actual, expected=expect)


if __name__ == "__main__":
    test_thread_group()
    test_elect_any_thread_group()
