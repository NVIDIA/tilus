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
import functools
from typing import Any, Callable, no_type_check

import numpy as np
import torch


@functools.cache
def _cuda_sleep_kernel():
    from hidet.ir.primitives.cuda.time import nano_sleep
    from hidet.lang import attrs, script, script_module
    from hidet.lang.types import int64

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
    kernel = _cuda_sleep_kernel()
    kernel(nanoseconds)


def benchmark_func(
    run_func: Callable[[], Any],
    warmup: int = 1,
    repeat: int = 5,
    clear_l2_cache: bool = True,
) -> float:
    num_bytes = 128 * 1024 * 1024
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
