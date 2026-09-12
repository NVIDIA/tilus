# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import tilus.lang.instantiated_script as instantiated_script_module
from tilus.ir.prog import Program
from tilus.lang.instantiated_script import JitInstance


class _DummyCompiledProgram:
    def __init__(self, program_dir: Path):
        self.program_dir = program_dir

    def get_launch_func(self):
        return self


def _jit_instance(cache_root: Path) -> JitInstance:
    instance = JitInstance.__new__(JitInstance)
    instance.frontend_fingerprint = "frontend"
    instance.backend_fingerprint = "backend"
    instance.specialization_cache_path = cache_root / "specializations" / "test.json"
    instance.cache_dir = cache_root / "scripts" / "test"
    instance.cache_dir_lock = instance.cache_dir / ".lock"
    instance.schedules = [{"block": 64}, {"block": 128}]
    instance.transpiled_schedules = [0, 1]
    instance.transpiled_programs = [Program.create({}), Program.create({})]
    instance.valid_schedules = [1]
    instance.valid_programs = [instance.transpiled_programs[1]]
    instance.compiled_programs = [_DummyCompiledProgram(cache_root / "programs" / "abc")]
    instance.dispatch_table = {}
    instance.load_dispatch_table = lambda: None
    return instance


def _empty_loaded_state(instance: JitInstance) -> None:
    instance.transpiled_schedules = []
    instance.transpiled_programs = []
    instance.valid_schedules = []
    instance.valid_programs = []
    instance.compiled_programs = []


def test_specialization_manifest_round_trip(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        instantiated_script_module.tilus.option,
        "get_option",
        lambda name: str(tmp_path),
    )
    original = _jit_instance(tmp_path)
    original.cache_dir.mkdir(parents=True)
    original.compiled_programs[0].program_dir.mkdir(parents=True)
    original._dump_specialization_cache()

    restored = _jit_instance(tmp_path)
    _empty_loaded_state(restored)

    assert restored._load_specialization_cache()
    assert restored.transpiled_schedules == [0, 1]
    assert len(restored.transpiled_programs) == 2
    assert restored.valid_schedules == []
    assert restored.compiled_programs == []


def test_backend_change_reuses_frontend_programs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        instantiated_script_module.tilus.option,
        "get_option",
        lambda name: str(tmp_path),
    )
    original = _jit_instance(tmp_path)
    original.cache_dir.mkdir(parents=True)
    original._dump_specialization_cache()

    restored = _jit_instance(tmp_path)
    _empty_loaded_state(restored)
    restored.backend_fingerprint = "different-backend"

    assert restored._load_specialization_cache()
    assert restored.transpiled_schedules == [0, 1]
    assert restored.valid_schedules == []
    assert restored.compiled_programs == []


def test_tuning_table_invalidates_after_backend_change(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(instantiated_script_module, "collect_tuning_metadata", lambda: {"gpu": "test"})
    original = _jit_instance(tmp_path)
    original.cache_dir.mkdir(parents=True)
    original.call_params = type("Params", (), {"tuning_params": [], "param_names": []})()
    original.dispatch_table = {(): 0}
    JitInstance.dump_dispatch_table(original)

    restored = _jit_instance(tmp_path)
    restored.backend_fingerprint = "new-backend"
    restored.dispatch_table = {(): 0}
    JitInstance.load_dispatch_table(restored)

    assert restored.dispatch_table == {}


@pytest.mark.parametrize("cached_indices", [(0, 1), (0,), (1,)])
def test_build_workers_receive_only_uncached_programs(monkeypatch, tmp_path: Path, cached_indices):
    from tilus.drivers import BuildOptions

    instance = _jit_instance(tmp_path)
    instance.cache_dir.mkdir(parents=True)
    instance.instance_name = "test"
    instance.build_options = BuildOptions()
    instance.script_cls = type("Script", (), {"debug_block": None})
    instance.failed_building = []
    instance.valid_schedules = []
    instance.valid_programs = []
    instance.compiled_programs = []
    paths = [tmp_path / "programs" / str(index) for index in range(2)]
    program_paths = {id(program): path for program, path in zip(instance.transpiled_programs, paths)}
    monkeypatch.setattr(instantiated_script_module, "lazy_init", lambda: None)
    monkeypatch.setattr(
        instantiated_script_module, "get_cache_dir", lambda program, options: program_paths[id(program)]
    )
    monkeypatch.setattr(
        instantiated_script_module, "compiled_program_exists", lambda path: path in [paths[i] for i in cached_indices]
    )
    monkeypatch.setattr(instantiated_script_module, "load_compiled_program", _DummyCompiledProgram)
    submitted = []

    def parallel(func, jobs):
        submitted.extend(program for program, options in jobs)
        return iter((True, str(program_paths[id(program)])) for program, options in jobs)

    monkeypatch.setattr(instantiated_script_module, "parallel_imap", parallel)
    instance._build_programs()

    assert submitted == [program for i, program in enumerate(instance.transpiled_programs) if i not in cached_indices]
    assert instance.valid_schedules == [0, 1]
    assert [compiled.program_dir for compiled in instance.compiled_programs] == paths
