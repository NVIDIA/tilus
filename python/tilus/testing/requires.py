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
from typing import Callable

import pytest

from tilus.target import Target, get_current_target, nvgpu_sm80, nvgpu_sm90, nvgpu_sm100


def _requires(target: Target) -> Callable[[Callable], Callable]:
    """
    Pytest fixture decorator that skips tests if the current GPU doesn't support the required architecture.

    Parameters
    ----------
    target : Target
        The required target architecture. Examples include 'sm_90a', 'sm_80',
    """

    def decorator(test_func):
        try:
            required_target = target
            current_target = get_current_target()
            current_capability = current_target.properties.compute_capability

            if not current_target.supports(required_target):
                return pytest.mark.skip(
                    f"Test requires architecture {required_target}, but current GPU capability is {current_capability}"
                )(test_func)
            return test_func
        except ValueError as e:
            # If we can't parse the architecture string, skip the test
            return pytest.mark.skip(f"Invalid architecture requirement: {e}")(test_func)
        except Exception as e:
            # If we can't determine current capability, skip the test
            return pytest.mark.skip(f"Cannot determine current GPU capability: {e}")(test_func)

    return decorator


class requires:
    nvgpu_sm90 = _requires(nvgpu_sm90)
    nvgpu_sm80 = _requires(nvgpu_sm80)
    nvgpu_sm100 = _requires(nvgpu_sm100)
