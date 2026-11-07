# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tilus.lang.instantiated_script as instantiated_script
import tilus.target
import torch
from tilus import int32
from tilus.lang.instantiated_script import InstantiatedScript
from tilus.target import Target, get_current_target, nvgpu_sm80, nvgpu_sm90, nvgpu_sm103a, scope
from tilus.testing import requires


def target_marker() -> int:
    """Encode the current target, evaluated while the script is transpiled."""
    major, minor = get_current_target().properties.compute_capability
    return major * 1000 + minor * 10 + (0 if get_current_target().properties.feature_suffix is None else 1)


class TargetExample(tilus.Script):
    def __call__(self, size: int32, out: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        marker = target_marker()
        g_out = self.global_view(out, dtype=int32, shape=[1])
        with self.thread_group(0, num_threads=1):
            self.store_global(g_out, self.register_tensor(dtype=int32, shape=[1], init=lambda i: marker), offsets=[0])


class CompileExample(tilus.Script):
    def __call__(self, out: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_out = self.global_view(out, dtype=int32, shape=[1])
        with self.thread_group(0, num_threads=1):
            self.store_global(g_out, self.register_tensor(dtype=int32, shape=[1], init=lambda i: 1), offsets=[0])


def test_jit_instance_per_target():
    script = InstantiatedScript(CompileExample, (), {})

    with scope(nvgpu_sm80):
        first = script._jit_instance_for(None)
    with scope(nvgpu_sm90):
        second = script._jit_instance_for(None)
    with scope(nvgpu_sm80):
        again = script._jit_instance_for(None)

    assert first is not second
    assert first is again


def test_compile_builds_each_target(monkeypatch):
    # Compiling needs nvcc but no GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    script = InstantiatedScript(CompileExample, (), {})

    with scope(nvgpu_sm80):
        first = script.compile(None)
    with scope(nvgpu_sm90):
        second = script.compile(None)

    first_dir = first.compiled_programs[0].program_dir
    second_dir = second.compiled_programs[0].program_dir
    assert first_dir != second_dir
    assert "sm80" in (first_dir / "options.txt").read_text()
    assert "sm90" in (second_dir / "options.txt").read_text()


def test_call_dispatches_on_target(monkeypatch):
    picks = []
    launches = []

    class FakeProgram:
        def __init__(self, target):
            self.target = target

        def get_launch_func(self):
            return lambda *args: launches.append(self.target)

    class FakeJitInstance:
        def __init__(self, *args):
            self.target = get_current_target()

        def _pick_best_program(self, args):
            picks.append((self.target, args[0]))
            return FakeProgram(self.target)

    monkeypatch.setattr(instantiated_script, "JitInstance", FakeJitInstance)
    script = InstantiatedScript(TargetExample, (), {})

    out = object()
    with scope(nvgpu_sm90):
        script(128, out)
        script(128, out)  # the last-launch shortcut
    with scope(nvgpu_sm103a):
        script(128, out)  # same arguments as the shortcut, another target
        script(256, out)
        script(128, out)  # the dispatch table of this target
    with scope(nvgpu_sm90):
        script(128, out)  # back to the first target, without building again

    assert launches == [nvgpu_sm90, nvgpu_sm90, nvgpu_sm103a, nvgpu_sm103a, nvgpu_sm103a, nvgpu_sm90]
    assert picks == [(nvgpu_sm90, 128), (nvgpu_sm103a, 128), (nvgpu_sm103a, 256)]


def compatible_target() -> Target:
    """Find another target whose binaries run on this GPU. A binary only runs on its own major version."""
    current = get_current_target()
    candidates = [
        target
        for target in vars(tilus.target).values()
        if isinstance(target, Target)
        and target != current
        and target.kind == current.kind
        and target.properties.feature_suffix is None
        and target.properties.compute_capability[0] == current.properties.compute_capability[0]
        and current.supports(target)
    ]
    if not candidates:
        pytest.skip("No other target can run on {}".format(current))
    return max(candidates, key=lambda target: target.properties.compute_capability)


@requires.nvgpu_sm80
def test_call_runs_binary_of_target():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script = InstantiatedScript(TargetExample, (), {})
    out = torch.zeros(1, dtype=torch.int32, device=device)

    script(128, out)
    assert out.item() == target_marker()
    expected = out.item()

    target = compatible_target()
    with scope(target):
        script(128, out)
        assert out.item() == target_marker()
        assert out.item() != expected

    script(128, out)
    assert out.item() == expected
