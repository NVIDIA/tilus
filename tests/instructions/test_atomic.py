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
"""End-to-end tests for ``self.atomic.*`` and non-atomic scatter stores.

Each test builds a small kernel, runs it on the GPU, and checks the result
against a reference computed with torch/python. Together they exercise:

- element-wise atomic add/max on global memory (no contention path)
- scatter atomic add on shared and global memory (contention path)
- DCE of unused output on scatter atomic add
- non-atomic scatter stores to shared and global memory
"""

import pytest
import tilus
import torch
from tilus import int32


class GlobalAtomicAddTile(tilus.Script):
    """Each thread contributes values[i] = i+1 to dst[i] via atomic add."""

    def __init__(self, n: int = 32):
        super().__init__()
        self.n = n

    def __call__(self, dst_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1  # 32 threads match n=32

        g = self.global_view(dst_ptr, dtype=int32, shape=[self.n])
        # values = [1, 2, ..., n], one element per thread under default spatial layout
        values = self.register_tensor(dtype=int32, shape=[self.n], init=lambda i: i + 1)  # type: ignore[arg-type, misc]
        self.atomic.global_add(g, values)


def test_atomic_global_add_no_contention():
    n = 32
    dst = torch.zeros(n, dtype=torch.int32, device="cuda")
    GlobalAtomicAddTile(n=n)(dst)
    expected = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(dst, expected)


class GlobalAtomicMaxTile(tilus.Script):
    """Each thread proposes max candidate thread_id to a pre-filled dst."""

    def __init__(self, n: int = 32):
        super().__init__()
        self.n = n

    def __call__(self, dst_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g = self.global_view(dst_ptr, dtype=int32, shape=[self.n])
        values = self.register_tensor(dtype=int32, shape=[self.n], init=lambda i: i)  # type: ignore[arg-type, misc]
        self.atomic.global_max(g, values)


def test_atomic_global_max_no_contention():
    n = 32
    # Start dst[i] = n - i. After atomicMax with candidate i, dst[i] = max(n-i, i).
    dst = torch.arange(n, 0, -1, dtype=torch.int32, device="cuda")
    expected = torch.maximum(dst.clone(), torch.arange(n, dtype=torch.int32, device="cuda"))
    GlobalAtomicMaxTile(n=n)(dst)
    torch.testing.assert_close(dst, expected)


class SharedScatterHistogram(tilus.Script):
    """Build a shared histogram with scatter-add then flush to global.

    Each of the 32 threads computes ``bin = thread_id % num_bins`` and scatters
    1 into that bin, so the expected histogram is
    ``[32 // num_bins] * num_bins`` (exact because 32 % num_bins == 0 here).

    The scatter atomic is called without consuming its return value; DCE will
    rewrite it to the destination-less ``red.*`` PTX form.
    """

    def __init__(self, num_threads: int = 32, num_bins: int = 4):
        super().__init__()
        self.num_threads = num_threads
        self.num_bins = num_bins

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = self.num_threads // 32

        bins = self.register_tensor(dtype=int32, shape=[self.num_threads], init=lambda i: i % self.num_bins)  # type: ignore[arg-type, misc]
        ones = self.register_tensor(dtype=int32, shape=[self.num_threads], init=1)

        s_hist = self.shared_tensor(dtype=int32, shape=[self.num_bins])
        # Seed shared memory from the caller-provided (zeroed) output buffer.
        # This sidesteps the missing register-layout rule for StoreSharedInst
        # when the producing register is an init-only AllocateRegister.
        g_out = self.global_view(out_ptr, dtype=int32, shape=[self.num_bins])
        init = self.load_global(g_out, offsets=[0], shape=[self.num_bins])
        self.store_shared(s_hist, init)
        self.sync()

        # Scatter-add 1 at bins[t] for each thread t.
        self.atomic.shared_scatter_add(s_hist, dim=0, indices=bins, values=ones)
        self.sync()

        # Flush shared histogram back to global.
        r_hist = self.load_shared(s_hist)
        self.store_global(g_out, r_hist, offsets=[0])


@pytest.mark.parametrize("num_bins", [4, 8, 16])
def test_atomic_shared_scatter_histogram(num_bins):
    num_threads = 32
    out = torch.zeros(num_bins, dtype=torch.int32, device="cuda")
    SharedScatterHistogram(num_threads=num_threads, num_bins=num_bins)(out)
    expected = torch.full((num_bins,), num_threads // num_bins, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(out, expected)


class GlobalScatterHistogram(tilus.Script):
    """Global-memory scatter histogram — same as shared but on global."""

    def __init__(self, num_threads: int = 32, num_bins: int = 4):
        super().__init__()
        self.num_threads = num_threads
        self.num_bins = num_bins

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = self.num_threads // 32

        bins = self.register_tensor(dtype=int32, shape=[self.num_threads], init=lambda i: i % self.num_bins)  # type: ignore[arg-type, misc]
        ones = self.register_tensor(dtype=int32, shape=[self.num_threads], init=1)

        g_hist = self.global_view(out_ptr, dtype=int32, shape=[self.num_bins])
        self.atomic.global_scatter_add(g_hist, dim=0, indices=bins, values=ones)


@pytest.mark.parametrize("num_bins", [4, 8])
def test_atomic_global_scatter_histogram(num_bins):
    num_threads = 32
    out = torch.zeros(num_bins, dtype=torch.int32, device="cuda")
    GlobalScatterHistogram(num_threads=num_threads, num_bins=num_bins)(out)
    expected = torch.full((num_bins,), num_threads // num_bins, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(out, expected)


class GlobalScatterPermutation(tilus.Script):
    """Permutation write via non-atomic scatter: dst[perm[t]] = values[t]."""

    def __init__(self, n: int = 32):
        super().__init__()
        self.n = n

    def __call__(self, dst_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1

        # dst[n-1-t] = t+100 for thread t.
        n = self.n
        perm = self.register_tensor(dtype=int32, shape=[n], init=lambda i: (n - 1) - i)  # type: ignore[arg-type, misc]
        values = self.register_tensor(dtype=int32, shape=[n], init=lambda i: i + 100)  # type: ignore[arg-type, misc]

        g = self.global_view(dst_ptr, dtype=int32, shape=[n])
        self.store_global_scatter(g, dim=0, indices=perm, values=values)


def test_store_global_scatter_permutation():
    n = 32
    dst = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    GlobalScatterPermutation(n=n)(dst)
    expected = torch.arange(100, 100 + n, dtype=torch.int32, device="cuda").flip(0)
    torch.testing.assert_close(dst, expected)


class SharedScatterRoundtrip(tilus.Script):
    """Scatter into shared then dump to global — tests store_shared_scatter."""

    def __init__(self, n: int = 32):
        super().__init__()
        self.n = n

    def __call__(self, dst_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1

        n = self.n
        perm = self.register_tensor(dtype=int32, shape=[n], init=lambda i: (n - 1) - i)  # type: ignore[arg-type, misc]
        values = self.register_tensor(dtype=int32, shape=[n], init=lambda i: i + 7)  # type: ignore[arg-type, misc]

        s = self.shared_tensor(dtype=int32, shape=[n])
        # Seed shared with the caller-provided (-1-filled) buffer so a scatter
        # miss would be observable downstream.
        g = self.global_view(dst_ptr, dtype=int32, shape=[n])
        fill = self.load_global(g, offsets=[0], shape=[n])
        self.store_shared(s, fill)
        self.sync()

        self.store_shared_scatter(s, dim=0, indices=perm, values=values)
        self.sync()

        r = self.load_shared(s)
        self.store_global(g, r, offsets=[0])


def test_store_shared_scatter_permutation():
    n = 32
    dst = torch.zeros(n, dtype=torch.int32, device="cuda")
    SharedScatterRoundtrip(n=n)(dst)
    expected = (torch.arange(n, dtype=torch.int32, device="cuda") + 7).flip(0)
    torch.testing.assert_close(dst, expected)


class SharedAtomicAddElementwise(tilus.Script):
    """Two disjoint warps both atomic-add into the same shared tile.

    Warp 0 contributes 1 per lane; warp 1 contributes 10 per lane. Expected
    final tile is [11]*32. Exercises the element-wise shared path across warps
    with real cross-thread contention on every address.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, dst_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 2

        s = self.shared_tensor(dtype=int32, shape=[32])
        # Seed shared from the caller-provided (zeroed) output buffer.
        g = self.global_view(dst_ptr, dtype=int32, shape=[32])
        with self.single_warp(warp=0):
            seed = self.load_global(g, offsets=[0], shape=[32])
            self.store_shared(s, seed)
        self.sync()

        with self.single_warp(warp=0):
            v = self.register_tensor(dtype=int32, shape=[32], init=1)
            self.atomic.shared_add(s, v, scope="cta")
        with self.single_warp(warp=1):
            v = self.register_tensor(dtype=int32, shape=[32], init=10)
            self.atomic.shared_add(s, v, scope="cta")
        self.sync()

        with self.single_warp(warp=0):
            r = self.load_shared(s)
            self.store_global(g, r, offsets=[0])


def test_atomic_shared_add_two_warps():
    dst = torch.zeros(32, dtype=torch.int32, device="cuda")
    SharedAtomicAddElementwise()(dst)
    expected = torch.full((32,), 11, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(dst, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
