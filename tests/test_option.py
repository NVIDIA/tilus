# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest
import tilus.option


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(128, ["git"]),
        FileNotFoundError(2, "No such file or directory", "git"),
    ],
)
def test_default_cache_dir_falls_back_to_home_without_git_repo(monkeypatch, tmp_path: Path, error):
    def failing_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", failing_run)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert tilus.option._get_default_cache_dir() == str(tmp_path / ".cache" / "tilus")


def _fake_toplevel(monkeypatch, toplevel: Path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=f"{toplevel}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_default_cache_dir_uses_repo_cache_in_tilus_repo(monkeypatch):
    repo_root = Path(tilus.option.__file__).resolve().parents[2]
    _fake_toplevel(monkeypatch, repo_root)

    assert tilus.option._get_default_cache_dir() == str(repo_root / ".cache")


def test_default_cache_dir_falls_back_to_home_in_other_repo(monkeypatch, tmp_path: Path):
    # e.g., tilus installed into a virtualenv that lives inside another project's git work tree
    _fake_toplevel(monkeypatch, tmp_path / "other-project")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert tilus.option._get_default_cache_dir() == str(tmp_path / ".cache" / "tilus")
