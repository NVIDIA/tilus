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
