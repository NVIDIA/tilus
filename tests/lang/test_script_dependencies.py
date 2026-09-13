# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import tilus
from tilus.lang.script_dependencies import script_dependency_fingerprint

_GLOBAL = {"value": 1}


class _Helper(tilus.Class):
    value = 1

    def method(self):
        return _GLOBAL["value"] + self.value


class _Base(tilus.Script):
    def inherited(self):
        return _GLOBAL["value"]


class _Script(_Base):
    def __init__(self, block=64):
        super().__init__()
        self.block = block

    def __call__(self, ptr: ~tilus.float16):
        helper = _Helper()
        value = helper.method() + self.inherited()
        tensor = self.register_tensor(tilus.float16, [self.block], init=value)
        self.store_global(ptr, tensor)


def _fingerprint(cls=_Script, inputs=()):
    return script_dependency_fingerprint(cls, inputs)


def test_tracks_external_globals(monkeypatch):
    original = _fingerprint()
    assert original is not None
    monkeypatch.setitem(_GLOBAL, "value", 2)
    assert _fingerprint() != original


def test_tracks_external_class_helpers(monkeypatch):
    original = _fingerprint()
    assert original is not None
    monkeypatch.setattr(_Helper, "value", 2)
    assert _fingerprint() != original


def test_tracks_inherited_methods(monkeypatch):
    original = _fingerprint()

    def replacement(self):
        return 2

    monkeypatch.setattr(_Base, "inherited", replacement)
    assert _fingerprint() is not None
    assert _fingerprint() != original


def _closed_script(value):
    class Closed(tilus.Script):
        def __call__(self):
            return value["value"]

    return Closed


def test_tracks_mutable_closure_values():
    state = {"value": 1}
    script = _closed_script(state)
    original = _fingerprint(script)
    assert original is not None
    state["value"] = 2
    assert _fingerprint(script) != original


def test_tracks_defaults_and_specialization_inputs(monkeypatch):
    original = _fingerprint()
    monkeypatch.setattr(_Script.__init__, "__defaults__", (128,))
    assert _fingerprint() != original
    assert _fingerprint(inputs=({"block": 64},)) != _fingerprint(inputs=({"block": 128},))


def test_rejects_opaque_values():
    assert _fingerprint(inputs=(object(),)) is None


def test_rejects_dynamic_dependencies():
    class Dynamic(tilus.Script):
        def __call__(self):
            return getattr(_Helper, "value")

    class Imports(tilus.Script):
        def __call__(self):
            import random

            return random.random()

    assert _fingerprint(Dynamic) is None
    assert _fingerprint(Imports) is None


def test_rejects_opaque_module_access():
    class Module(tilus.Script):
        def __call__(self):
            return sys.version

    # Passing a module itself cannot statically enumerate its consumers.
    class PassedModule(tilus.Script):
        def __call__(self):
            return str(sys)

    assert _fingerprint(Module) is not None
    assert _fingerprint(PassedModule) is None


def test_matmul_dependencies_are_supported():
    path = Path(__file__).parents[1] / "kernels/matmul/test_matmul_v2.py"
    spec = importlib.util.spec_from_file_location("dependency_test_matmul", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert _fingerprint(module.MatmulV2) is not None
    finally:
        del sys.modules[spec.name]


def test_fingerprint_is_stable_across_processes(tmp_path):
    path = tmp_path / "script.py"
    path.write_text(
        "import tilus\n"
        "from tilus.lang.script_dependencies import script_dependency_fingerprint\n"
        "VALUE = 1\n"
        "class Kernel(tilus.Script):\n"
        "    def __call__(self):\n"
        "        return VALUE\n"
        "print(script_dependency_fingerprint(Kernel, ()))\n"
    )
    first = subprocess.check_output([sys.executable, str(path)], text=True).strip()
    second = subprocess.check_output([sys.executable, str(path)], text=True).strip()
    assert first != "None"
    assert first == second
    path.write_text(path.read_text().replace("VALUE = 1", "VALUE = 2"))
    assert subprocess.check_output([sys.executable, str(path)], text=True).strip() != first


_ANNOTATION = tilus.int32


def test_tracks_string_annotations_that_shadow_local_names(monkeypatch):
    class Annotated(tilus.Script):
        def __call__(self, value: "_ANNOTATION"):
            _ANNOTATION = 42
            return value + _ANNOTATION

    original = _fingerprint(Annotated)
    assert original is not None
    monkeypatch.setitem(globals(), "_ANNOTATION", tilus.float32)
    assert _fingerprint(Annotated) != original


def test_tracks_helper_function_globals(monkeypatch):
    def helper():
        return _GLOBAL["value"]

    class Kernel(tilus.Script):
        def __call__(self):
            return helper()

    original = _fingerprint(Kernel)
    assert original is not None
    monkeypatch.setitem(_GLOBAL, "value", 2)
    assert _fingerprint(Kernel) != original


def test_tracks_method_binding_kind(monkeypatch):
    class Helper(tilus.Class):
        @staticmethod
        def value(cls=None):
            return cls is None

    class Kernel(tilus.Script):
        def __call__(self):
            return Helper.value()

    original = _fingerprint(Kernel)
    assert original is not None
    method = vars(Helper)["value"].__func__
    monkeypatch.setattr(Helper, "value", classmethod(method))
    assert _fingerprint(Kernel) != original
