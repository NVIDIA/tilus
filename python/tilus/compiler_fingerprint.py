"""Content fingerprints for Tilus compiler cache stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import filelock

FRONTEND_CACHE_VERSION = "2"
BACKEND_CACHE_VERSION = "2"

_INDEX_VERSION = 1
_PACKAGE_ROOT = Path(__file__).parent
_SOURCE_SUFFIXES = {".py", ".h", ".cuh", ".cc", ".cpp"}


def _stat_key(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _load_index(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("version") == _INDEX_VERSION and isinstance(data.get("files"), dict):
            return data
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {"version": _INDEX_VERSION, "files": {}}


def _dump_index(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=True)
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _leaf_digest(path: Path, cached: Any, stat_key: list[int] | None = None) -> tuple[str, list[int]]:
    for _ in range(3):
        stat_key = _stat_key(path) if stat_key is None else stat_key
        if isinstance(cached, dict) and cached.get("stat") == stat_key and isinstance(cached.get("digest"), str):
            try:
                if len(bytes.fromhex(cached["digest"])) == 32:
                    return cached["digest"], stat_key
            except ValueError:
                pass

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if _stat_key(path) == stat_key:
            return digest, stat_key
        stat_key = None
    raise RuntimeError(f"Compiler source changed repeatedly while fingerprinting: {path}")


def _fingerprint_index_path(cache_root: str | Path | None = None) -> Path:
    if cache_root is None:
        import tilus.option

        cache_root = tilus.option.get_option("cache_dir")
    return Path(cache_root) / "compiler" / f"source-hashes-v{_INDEX_VERSION}.json"


def _content_fingerprints(
    stages: Iterable[tuple[str, Iterable[str]]],
    *,
    package_root: Path = _PACKAGE_ROOT,
    index_path: Path | None = None,
) -> tuple[str, ...]:
    """Validate leaves once and reuse persisted Merkle nodes for unchanged subtrees.

    Directory timestamps cannot detect edits to existing files, so every process
    still stats source leaves. scandir avoids walking bytecode directories and
    redundant path probes; both compiler stages share the same traversal.
    """
    package_root = package_root.resolve()
    stages = [(version, tuple(roots)) for version, roots in stages]
    index_path = _fingerprint_index_path() if index_path is None else index_path
    root_key = hashlib.sha256(str(package_root).encode()).hexdigest()[:16]

    def calculate(index: dict[str, Any]) -> tuple[tuple[str, ...], bool]:
        records = index["files"]
        trees = index.setdefault("trees", {})
        if not isinstance(trees, dict):
            trees = index["trees"] = {}
        visited: dict[str, str] = {}
        changed = False

        def leaf(path: Path, relative: str, stat: os.stat_result) -> str:
            nonlocal changed
            record_key = f"{root_key}:{relative}"
            old_record = records.get(record_key)
            stat_key = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            digest, stat_key = _leaf_digest(path, old_record, stat_key)
            new_record = {"stat": stat_key, "digest": digest}
            if old_record != new_record:
                records[record_key] = new_record
                changed = True
            visited[relative] = digest
            return digest

        def directory(path: Path, relative: str) -> str:
            nonlocal changed
            if relative in visited:
                return visited[relative]
            children = []
            with os.scandir(path) as entries:
                for entry in entries:
                    child_relative = f"{relative}/{entry.name}"
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name == "__pycache__":
                            continue
                        digest = directory(Path(entry.path), child_relative)
                        children.append([entry.name + "/", digest])
                    elif os.path.splitext(entry.name)[1] in _SOURCE_SUFFIXES and entry.is_file():
                        digest = leaf(Path(entry.path), child_relative, entry.stat())
                        children.append([entry.name, digest])
            children.sort()
            record_key = f"{root_key}:{relative}"
            old_record = trees.get(record_key)
            if isinstance(old_record, dict) and old_record.get("children") == children:
                digest = old_record.get("digest")
                if isinstance(digest, str):
                    try:
                        if len(bytes.fromhex(digest)) == 32:
                            visited[relative] = digest
                            return digest
                    except ValueError:
                        pass
            node = hashlib.sha256(b"directory\0")
            for name, digest in children:
                node.update(name.encode())
                node.update(b"\0")
                node.update(bytes.fromhex(digest))
            digest = node.hexdigest()
            trees[record_key] = {"children": children, "digest": digest}
            changed = True
            visited[relative] = digest
            return digest

        fingerprints = []
        for version, roots in stages:
            stage = hashlib.sha256(version.encode())
            for relative in sorted(roots):
                if relative in visited:
                    digest = visited[relative]
                else:
                    path = package_root / relative
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        continue
                    if path.is_dir():
                        digest = directory(path, relative)
                    else:
                        digest = leaf(path, relative, stat)
                stage.update(relative.encode())
                stage.update(b"\0")
                stage.update(bytes.fromhex(digest))
            fingerprints.append(stage.hexdigest()[:16])
        return tuple(fingerprints), changed

    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(str(index_path) + ".lock"):
            index = _load_index(index_path)
            fingerprints, changed = calculate(index)
            if changed:
                _dump_index(index_path, index)
            return fingerprints
    except OSError:
        return calculate({"version": _INDEX_VERSION, "files": {}})[0]


def _content_fingerprint(
    version: str,
    relative_roots: Iterable[str],
    *,
    package_root: Path = _PACKAGE_ROOT,
    index_path: Path | None = None,
) -> str:
    return _content_fingerprints([(version, relative_roots)], package_root=package_root, index_path=index_path)[0]


@lru_cache(maxsize=8)
def _compiler_fingerprints(cache_root: str | Path | None = None) -> tuple[str, str]:
    frontend_roots = [
        "target.py",
        "__init__.py",
        "option.py",
        "ir",
        "hidet",
        "lang",
        "utils",
        "compiler_fingerprint.py",
    ]
    backend_roots = ["backends", "transforms", "hidet", "drivers.py", "compiler_fingerprint.py"]
    # Visit backend roots first so the frontend reuses the bundled Hidet node.
    backend, frontend = _content_fingerprints(
        [(BACKEND_CACHE_VERSION, backend_roots), (FRONTEND_CACHE_VERSION, frontend_roots)],
        index_path=_fingerprint_index_path(cache_root),
    )
    return frontend, backend


@lru_cache(maxsize=8)
def frontend_fingerprint(cache_root: str | Path | None = None) -> str:
    """Fingerprint code that converts a Script specialization to a Tilus Program."""
    return _compiler_fingerprints(cache_root)[0]


@lru_cache(maxsize=8)
def backend_fingerprint(cache_root: str | Path | None = None) -> str:
    """Fingerprint code and headers that convert a Tilus Program to a binary."""
    return _compiler_fingerprints(cache_root)[1]
