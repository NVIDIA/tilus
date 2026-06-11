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
    _tilus_version,
    collect_tuning_metadata,
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
