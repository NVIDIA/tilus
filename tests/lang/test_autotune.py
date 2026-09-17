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
"""Tests for the autotune schedule-space generation and the hardware-aware dispatch cache.

These tests exercise pure-Python autotuner logic and do not require a GPU.
"""

from typing import Any, Sequence

import tilus
from tilus import float16, int32
from tilus.lang.instantiated_script import (
    InstantiatedScript,
    _make_tuple_getter,
    _tilus_version,
    _tuning_key_part,
    collect_tuning_metadata,
    construct_keys,
    extract_keys,
    generate_schedules,
    span_space,
    tuning_metadata_matches,
)


def test_span_space_single_and_grouped_keys():
    space: dict[str, Sequence[Any]] = {"m": [1, 2, 3], "n, k": [[1, 2], [3, 4]]}
    spanned = span_space(space)
    # 3 choices for m, 2 choices for (n, k) -> 6 combinations
    assert len(spanned) == 6
    assert {"m": 1, "n": 1, "k": 2} in spanned
    assert {"m": 3, "n": 3, "k": 4} in spanned
    # every spanned schedule has the flattened keys
    for entry in spanned:
        assert set(entry) == {"m", "n", "k"}


@tilus.autotune("block_m, block_n", [[64, 64], [128, 64], [128, 128]])
@tilus.autotune("block_k", [16, 32, 64])
class _DummyMatmul(tilus.Script):
    def __init__(self, block_m: int, block_n: int, block_k: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def __call__(self, m: int32, a_ptr: ~float16):  # pragma: no cover - never executed
        pass


def _dummy_schedules():
    space = getattr(_DummyMatmul, "_autotune_space")
    return generate_schedules(space, _DummyMatmul, script_args=(), script_kwargs={})


def test_generate_schedules_cartesian_product():
    schedules = _dummy_schedules()
    # 3 (block_m, block_n) x 3 (block_k) = 9 schedules
    assert len(schedules) == 9
    assert {"block_m": 64, "block_n": 64, "block_k": 16} in schedules
    assert {"block_m": 128, "block_n": 128, "block_k": 64} in schedules


def test_tuning_key_parts_are_reused():
    _tuning_key_part.cache_clear()
    args = (128, 257, 7)
    expected = extract_keys(args, const_params=[2], tuning_params=[0, 1])
    assert expected == construct_keys(const_params=[7], tuning_params=[128, 257])
    first = _tuning_key_part.cache_info()
    extract_keys(args, const_params=[2], tuning_params=[0, 1])
    second = _tuning_key_part.cache_info()
    assert second.hits == first.hits + 2


def test_tuple_getter_selects_positional_arguments():
    args = ("a", "b", "c")
    assert _make_tuple_getter([])(args) == ()
    assert _make_tuple_getter([1])(args) == ("b",)
    assert _make_tuple_getter([2, 0])(args) == ("c", "a")


def test_repeated_shape_uses_last_dispatch_entry():
    calls = []

    def launch(*args):
        calls.append(args)

    script = InstantiatedScript.__new__(InstantiatedScript)
    script.with_default = False
    script.params = type("Params", (), {"param_names": ["ptr", "size", "mode"]})()
    script.const_params = [2]
    script.tuning_params = [1]
    script.kernel_params = [0, 1]
    script.jit_instances = {}
    script._dispatch_arg_getter = _make_tuple_getter([2, 1])
    script._kernel_arg_getter = _make_tuple_getter([0, 1])
    script._last_dispatch_args = None
    script._last_compiled_func = None

    ptr = object()
    keys = extract_keys((ptr, 128, 7), script.const_params, script.tuning_params)
    script.dispatch_table = {keys: launch}  # type: ignore[dict-item]
    script(ptr, 128, 7)
    script.dispatch_table.clear()
    script(ptr, 128, 7)

    assert calls == [(ptr, 128), (ptr, 128)]


# ---------------------------------------------------------------------------
# Hardware-aware dispatch-cache fingerprint
# ---------------------------------------------------------------------------


def test_collect_tuning_metadata_has_expected_keys():
    meta = collect_tuning_metadata()
    assert set(meta) == {"tilus_version", "target", "gpu", "compute_capability", "cuda_version"}
    # all values must be strings (never None) so they serialize and compare cleanly
    assert all(isinstance(v, str) for v in meta.values())


def test_tuning_version_is_release_base():
    # The fingerprint must key on the release base version (e.g. "0.2.1"), not the full SCM/dev
    # version ("0.2.1.dev19+g<hash>"), so dev builds off the same release keep sharing the cache.
    version = _tilus_version()
    assert ".dev" not in version
    assert "+" not in version


def test_tuning_metadata_matches_identical():
    meta = {"gpu": "NVIDIA B300", "compute_capability": "10.3", "cuda_version": "13.0"}
    assert tuning_metadata_matches(meta, meta)


def test_tuning_metadata_matches_detects_gpu_mismatch():
    saved = {"gpu": "NVIDIA B200", "compute_capability": "10.0"}
    current = {"gpu": "NVIDIA B300", "compute_capability": "10.3"}
    assert not tuning_metadata_matches(saved, current)


def test_tuning_metadata_matches_wildcard():
    saved = {"gpu": "*", "compute_capability": "10.3"}
    current = {"gpu": "NVIDIA B300", "compute_capability": "10.3"}
    assert tuning_metadata_matches(saved, current)


def test_tuning_metadata_matches_rejects_legacy_or_missing():
    current = {"gpu": "NVIDIA B300"}
    # legacy cache files without a metadata mapping must never match
    assert not tuning_metadata_matches(None, current)
    assert not tuning_metadata_matches([], current)
    # a metadata block missing a required field does not match
    assert not tuning_metadata_matches({}, current)
