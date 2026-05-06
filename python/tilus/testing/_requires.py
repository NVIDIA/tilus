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
from typing import Any, Callable

from tilus.target import Target, get_current_target, nvgpu_sm80, nvgpu_sm90, nvgpu_sm100, nvgpu_sm100a, scope


class _CompileOnlyDone(Exception):
    """Raised inside a compile-only test to short-circuit execution after a successful compile."""


def _requires(target: Target) -> Callable[[Callable], Callable]:
    """
    Pytest decorator that adapts test behavior to the current GPU.

    If the current GPU supports the required target, the test runs unchanged.

    Otherwise, the test runs in *compile-only* mode:
    - The current compilation target is overridden to ``target`` for the duration of the test.
    - The first ``InstantiatedScript.__call__`` invocation is redirected to
      :py:meth:`InstantiatedScript.compile <tilus.InstantiatedScript.compile>`, which transpiles +
      builds every schedule in the autotune space without running the kernel.
    - After the compile succeeds, a sentinel exception is raised to short-circuit the rest of the
      test body; the decorator catches the sentinel and treats the test as passed.

    Parameters
    ----------
    target : Target
        The required target architecture, e.g. ``nvgpu_sm100a``.
    """

    def decorator(test_func: Callable) -> Callable:
        try:
            current_target = get_current_target()
            supports_target = current_target.supports(target)
        except Exception:
            # Could not determine the current target (e.g. no GPU available).
            # Fall through to compile-only mode -- compilation does not need a runtime GPU.
            supports_target = False

        if supports_target:
            return test_func

        @functools.wraps(test_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Importing here avoids a top-level circular import: tilus.testing is imported eagerly
            # by tests but tilus.lang.instantiated_script depends on the rest of the package.
            from tilus.lang.instantiated_script import InstantiatedScript

            original_call = InstantiatedScript.__call__

            def compile_only_call(self: InstantiatedScript, *call_args: Any, **call_kwargs: Any) -> Any:
                self.compile(*call_args, **call_kwargs)
                raise _CompileOnlyDone()

            InstantiatedScript.__call__ = compile_only_call  # type: ignore[method-assign]
            try:
                with scope(target):
                    test_func(*args, **kwargs)
            except _CompileOnlyDone:
                pass
            finally:
                InstantiatedScript.__call__ = original_call  # type: ignore[method-assign]

        return wrapper

    return decorator


class requires:
    nvgpu_sm90 = _requires(nvgpu_sm90)
    nvgpu_sm80 = _requires(nvgpu_sm80)
    nvgpu_sm100 = _requires(nvgpu_sm100)
    nvgpu_sm100a = _requires(nvgpu_sm100a)
