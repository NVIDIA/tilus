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
import functools
from typing import Any, Callable, no_type_check

import numpy as np
import torch


@functools.cache
def _cuda_sleep_kernel():
    from tilus.hidet.ir.primitives.cuda.time import nano_sleep
    from tilus.hidet.lang import attrs, script, script_module
    from tilus.hidet.lang.types import int64

    with script_module() as module:

        @no_type_check
        @script
        def cuda_sleep_kernel(nanoseconds: int64):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            # since the nano_sleep has a upper bound to sleep, approximately 1 millisecond, we break the given
            # nanoseconds into multiple milliseconds
            for _ in range(nanoseconds // 1000000):
                nano_sleep(1000000)

            nano_sleep(nanoseconds % 1000000)

    return module.build()


def cuda_sleep(nanoseconds: int) -> None:
    """A sleep cuda kernel that will sleep for given nanoseconds."""
    # Convert nanoseconds to CUDA clock cycles (approximate: 1 GHz = 1 cycle/ns)
    # torch.cuda._sleep takes cycles, not nanoseconds
    torch.cuda._sleep(nanoseconds)


@functools.lru_cache(maxsize=None)
def _l2_clear_nbytes(device_index: int) -> int:
    """Return the number of bytes to write in order to evict the L2 cache on the given device.

    Writing a slab twice the device's L2 cache size reliably flushes it, so that each benchmarked
    iteration starts from a cold L2. The actual L2 size is queried from the device (it varies across
    architectures); a 128 MiB floor preserves the previous behavior when the size cannot be determined.
    """
    floor = 128 * 1024 * 1024
    try:
        l2_size = torch.cuda.get_device_properties(device_index).L2_cache_size
    except Exception:
        l2_size = 0
    return max(2 * int(l2_size), floor)


def benchmark_func(
    run_func: Callable[[], Any],
    warmup: int = 1,
    repeat: int = 5,
    clear_l2_cache: bool = True,
) -> float:
    num_bytes = _l2_clear_nbytes(torch.cuda.current_device())
    memory_slab = torch.empty(num_bytes, dtype=torch.int8, device="cuda")

    assert repeat >= 1

    events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * (repeat + warmup))]

    # initialize events and the sleep kernel
    for event in events:
        event.record()
    memory_slab[:] = 0
    cuda_sleep(0)

    # warmup and benchmark
    torch.cuda.synchronize()
    for i in range(warmup + repeat):
        if clear_l2_cache:
            memory_slab[:] = 0
        if i == warmup:
            # from this iteration, we start to runs that will count the time
            cuda_sleep(repeat * 150000)  # sleep 150 microseconds for each kernel launch
        events[i * 2].record()
        run_func()
        events[i * 2 + 1].record()
    torch.cuda.synchronize()
    results = [events[i * 2].elapsed_time(events[i * 2 + 1]) for i in range(warmup, warmup + repeat)]

    return float(np.median(results))
