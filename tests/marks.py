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
"""Custom pytest markers for tilus test suite.

Usage:
    @pytest.mark.gpu
    def test_something_requiring_gpu():
        ...

    @pytest.mark.slow
    def test_long_running():
        ...

To run only GPU tests:
    pytest -m gpu

To skip slow tests:
    pytest -m "not slow"
"""
import functools

import pytest

# Re-export common markers for convenience
gpu = pytest.mark.gpu
slow = pytest.mark.slow
ampere = pytest.mark.ampere
hopper = pytest.mark.hopper
blackwell = pytest.mark.blackwell


def requires_compute_capability(major, minor=0):
    """Skip test if GPU compute capability is below the specified version.

    Parameters
    ----------
    major : int
        Major compute capability version (e.g. 8 for Ampere).
    minor : int, optional
        Minor compute capability version, by default 0.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return pytest.mark.skip(reason="CUDA not available")
        cap = torch.cuda.get_device_capability(0)
        if cap < (major, minor):
            return pytest.mark.skip(reason=f"Requires compute capability >= {major}.{minor}, got {cap[0]}.{cap[1]}")
    except ImportError:
        return pytest.mark.skip(reason="torch not installed")
    return pytest.mark.gpu


def parametrize_dtypes(*dtypes):
    """Parametrize a test over multiple data types.

    Parameters
    ----------
    *dtypes
        Data type strings to test (e.g. "float16", "float32", "bfloat16").
    """
    return pytest.mark.parametrize("dtype", dtypes, ids=[str(d) for d in dtypes])
