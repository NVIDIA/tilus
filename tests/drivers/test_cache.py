# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import tilus
from tilus.drivers import BuildOptions, get_cache_dir
from tilus.ir.prog import Program
from tilus.target import nvgpu_sm90


def test_get_cache_dir_tracks_ambient_cache_root(monkeypatch, tmp_path: Path):
    roots = iter([tmp_path / "first", tmp_path / "second"])
    monkeypatch.setattr(
        tilus.option,
        "get_option",
        lambda name: str(next(roots)) if name == "cache_dir" else False,
    )
    monkeypatch.setattr(tilus.target, "get_current_target", lambda: nvgpu_sm90)

    program = Program.create({})
    first = get_cache_dir(program, BuildOptions())
    second = get_cache_dir(program, BuildOptions())

    assert first.parent.parent == tmp_path / "first"
    assert second.parent.parent == tmp_path / "second"


def test_get_cache_dir_tracks_backend_fingerprint(monkeypatch, tmp_path: Path):
    fingerprints = iter(["backend-a", "backend-b"])
    monkeypatch.setattr(
        tilus.option,
        "get_option",
        lambda name: str(tmp_path) if name == "cache_dir" else False,
    )
    monkeypatch.setattr(tilus.target, "get_current_target", lambda: nvgpu_sm90)
    monkeypatch.setattr("tilus.drivers.backend_fingerprint", lambda cache_root: next(fingerprints))

    program = Program.create({})
    first = get_cache_dir(program, BuildOptions())
    second = get_cache_dir(program, BuildOptions())

    assert first != second
