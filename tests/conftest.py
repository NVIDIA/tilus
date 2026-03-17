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
import os
import subprocess
from pathlib import Path

import hidet
import pytest
import tilus
import tilus.utils


def _collect_gpu_diagnostics():
    """Collect GPU and CUDA environment info for CI diagnostic reports.

    Gathers hardware and driver details so test failures can be
    correlated with the specific GPU environment. Results are printed
    to stdout for inclusion in CI logs.
    """
    diag = []

    # GPU hardware info
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                diag.append(f"GPU: {line.strip()}")
        else:
            diag.append("GPU: nvidia-smi returned non-zero")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        diag.append("GPU: nvidia-smi not available")

    # CUDA compiler version
    try:
        proc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        for line in proc.stdout.split("\n"):
            if "release" in line.lower():
                diag.append(f"CUDA: {line.strip()}")
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        diag.append("CUDA: nvcc not found")

    # CUDA runtime libraries
    try:
        proc = subprocess.run(
            ["bash", "-c", "ldconfig -p 2>/dev/null | grep -c libcuda"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        diag.append(f"libcuda entries: {proc.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Query compute capability from nvidia-smi
    try:
        proc = subprocess.run(
            ["bash", "-c", "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.stdout.strip():
            diag.append(f"Compute capability: {proc.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check cuda-python bindings
    try:
        import cuda

        diag.append(f"cuda-python: {cuda.__version__}")
    except ImportError:
        diag.append("cuda-python: not installed")

    # Check torch CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            diag.append(f"torch.cuda: {torch.version.cuda} (device count: {torch.cuda.device_count()})")
        else:
            diag.append("torch.cuda: not available")
    except ImportError:
        diag.append("torch: not installed")

    if diag:
        print("\n[tilus-test] GPU environment diagnostics:")
        for d in diag:
            print(f"  {d}")


def pytest_sessionstart(session):
    """Initialize tilus options and collect GPU diagnostics before tests.

    Called after the Session object has been created and before performing
    collection and entering the run test loop.
    """
    # set the cache directory to a subdirectory of the current directory
    tilus.option.cache_dir(Path(tilus.option.get_option("cache_dir")) / ".test_cache")
    # we do not clear the cache here since vscode may run tests in parallel
    # print("Cache directory: {}".format(hidet.option.get_cache_dir()))
    # tilus.utils.clear_cache()

    # collect GPU diagnostics when running in CI
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        _collect_gpu_diagnostics()


@pytest.fixture(autouse=True)
def clear_before_test():
    """Clear the memory cache before each test."""
    import gc

    import torch

    torch.cuda.empty_cache()
    if hidet.cuda.available():
        hidet.runtime.storage.current_memory_pool("cuda").clear()
    gc.collect()  # release resources with circular references but are unreachable
    yield
    # run after each test
    pass
