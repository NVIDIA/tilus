# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from tilus.compiler_fingerprint import _content_fingerprint


def fingerprint(package_root: Path, index_path: Path, version: str = "1") -> str:
    return _content_fingerprint(version, ["frontend"], package_root=package_root, index_path=index_path)


def test_fingerprint_reuses_unchanged_leaf_hashes(monkeypatch, tmp_path: Path):
    package_root = tmp_path / "tilus"
    source_path = package_root / "frontend" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("old source")
    index_path = tmp_path / "cache" / "index.json"

    expected = fingerprint(package_root, index_path)
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError(self)))

    assert fingerprint(package_root, index_path) == expected


def test_fingerprint_changes_with_content_and_stage_version(tmp_path: Path):
    package_root = tmp_path / "tilus"
    source_path = package_root / "frontend" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("old source")
    index_path = tmp_path / "cache" / "index.json"

    original = fingerprint(package_root, index_path)
    source_path.write_text("new source with a different size")

    assert fingerprint(package_root, index_path) != original
    assert fingerprint(package_root, index_path, version="2") != fingerprint(package_root, index_path, version="1")


def test_fingerprint_recovers_from_corrupt_index(tmp_path: Path):
    package_root = tmp_path / "tilus"
    source_path = package_root / "frontend" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source")
    index_path = tmp_path / "cache" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("not json")

    result = fingerprint(package_root, index_path)

    assert result == fingerprint(package_root, index_path)
    assert json.loads(index_path.read_text())["version"] == 1


def test_fingerprint_tracks_added_removed_and_renamed_files(tmp_path: Path):
    package_root = tmp_path / "tilus"
    source_path = package_root / "frontend" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source")
    index_path = tmp_path / "cache" / "index.json"
    original = fingerprint(package_root, index_path)

    second_path = source_path.with_name("second.py")
    second_path.write_text("second")
    assert fingerprint(package_root, index_path) != original
    second_path.unlink()
    assert fingerprint(package_root, index_path) == original
    source_path.rename(source_path.with_name("renamed.py"))
    assert fingerprint(package_root, index_path) != original


def test_fingerprint_recovers_from_non_mapping_index(tmp_path: Path):
    package_root = tmp_path / "tilus"
    source_path = package_root / "frontend" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source")
    index_path = tmp_path / "cache" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("[]")

    assert fingerprint(package_root, index_path) == fingerprint(package_root, index_path)


def test_merkle_edit_rehashes_only_changed_branch(monkeypatch, tmp_path: Path):
    import tilus.compiler_fingerprint as module

    package_root = tmp_path / "tilus"
    left = package_root / "frontend" / "left" / "source.py"
    right = package_root / "frontend" / "right" / "source.py"
    for path in [left, right]:
        path.parent.mkdir(parents=True)
        path.write_text("old source")
    index_path = tmp_path / "cache" / "index.json"
    original = fingerprint(package_root, index_path)
    directory_mtime = left.parent.stat().st_mtime_ns
    left.write_text("new source")
    assert left.parent.stat().st_mtime_ns == directory_mtime

    hashed_nodes = []
    sha256 = module.hashlib.sha256

    def record_hash(data=b""):
        if data == b"directory\0":
            hashed_nodes.append(data)
        return sha256(data)

    read_paths = []
    read_bytes = Path.read_bytes

    def record_read(path):
        read_paths.append(path)
        return read_bytes(path)

    monkeypatch.setattr(module.hashlib, "sha256", record_hash)
    monkeypatch.setattr(Path, "read_bytes", record_read)
    assert fingerprint(package_root, index_path) != original
    assert read_paths == [left]
    assert len(hashed_nodes) == 2  # changed left node and its frontend ancestor


def test_compiler_stages_share_leaf_validation(monkeypatch, tmp_path: Path):
    from tilus.compiler_fingerprint import _content_fingerprints

    package_root = tmp_path / "tilus"
    source = package_root / "frontend" / "nested" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("source")
    read_paths = []
    read_bytes = Path.read_bytes

    def record_read(path):
        read_paths.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    stages = [("1", ["frontend"]), ("1", ["frontend/nested"])]
    result = _content_fingerprints(stages, package_root=package_root, index_path=tmp_path / "index.json")
    assert len(result) == 2
    assert read_paths == [source]


def test_frontend_fingerprints_all_language_modules(monkeypatch, tmp_path: Path):
    import tilus.compiler_fingerprint as module

    content_fingerprints = module._content_fingerprints

    def fingerprint_test_package(stages, **kwargs):
        return content_fingerprints(stages, package_root=tmp_path / "tilus", **kwargs)

    monkeypatch.setattr(module, "_content_fingerprints", fingerprint_test_package)

    def frontend():
        module._compiler_fingerprints.cache_clear()  # simulate a new process
        return module._compiler_fingerprints(str(tmp_path))[0]

    try:
        for relative in ["lang/methods/register_tensor.py", "lang/constructs/loops.py", "hidet/lang/transpiler.py"]:
            source = tmp_path / "tilus" / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("old source")
            original = frontend()
            source.write_text("changed frontend semantics")
            assert frontend() != original
    finally:
        module._compiler_fingerprints.cache_clear()


def test_backend_tracks_shared_helpers(monkeypatch, tmp_path):
    import tilus.compiler_fingerprint as module

    content_fingerprints = module._content_fingerprints

    def test_package(stages, **kwargs):
        return content_fingerprints(stages, package_root=tmp_path / "tilus", **kwargs)

    monkeypatch.setattr(module, "_content_fingerprints", test_package)

    def backend():
        module._compiler_fingerprints.cache_clear()
        return module._compiler_fingerprints(str(tmp_path))[1]

    try:
        for relative in ["ir/utils/thread_group_stack.py", "utils/py.py", "target.py", "option.py"]:
            source = tmp_path / "tilus" / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("old backend helper")
            original = backend()
            source.write_text("changed backend semantics")
            assert backend() != original
    finally:
        module._compiler_fingerprints.cache_clear()
