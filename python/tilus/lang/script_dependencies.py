# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Conservative snapshots of Python dependencies consumed by transpilation."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import marshal
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import numpy as np

from tilus.hidet.ir.type import BaseType
from tilus.lang.instructions.base import InstructionGroup
from tilus.lang.script import Attributes

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "object",
        "pow",
        "range",
        "reversed",
        "round",
        "slice",
        "str",
        "sum",
        "super",
        "tuple",
        "zip",
    )
}


class _UntrackedDependency(Exception):
    pass


def script_dependency_fingerprint(script_cls: type, inputs: Any) -> str | None:
    """Return a deterministic dependency digest, or decline persistent reuse.

    User functions and classes include their source, code, defaults, attributes,
    referenced globals, and closure cells. Module access must resolve statically
    to tracked attributes; opaque objects, imports, and reflective builtins fall
    back to fresh transpilation. Compiler-owned types are covered by the frontend
    fingerprint, while dtype instances also snapshot their runtime fields.
    """
    seen: dict[int, tuple[int, Any]] = {}

    def compiler_owned(value: Any) -> bool:
        try:
            path = Path(inspect.getfile(value)).resolve()
        except (OSError, TypeError):
            return False
        return path.is_relative_to(_PACKAGE_ROOT)

    def snapshot(value: Any) -> Any:
        if type(value) in (type(None), bool, int, float, str, bytes):
            return [type(value).__name__, repr(value)]
        if isinstance(value, np.generic) and type(value).__module__ == "numpy" and not value.dtype.hasobject:
            return ["numpy_scalar", value.dtype.str, value.tobytes().hex()]
        for name, builtin in _SAFE_BUILTINS.items():
            if value is builtin:
                return ["builtin", name]
        if id(value) in seen:
            return ["reference", seen[id(value)][0]]
        seen[id(value)] = (len(seen), value)
        if type(value) in (tuple, list):
            return [type(value).__name__, [snapshot(item) for item in value]]
        if type(value) is dict:
            return ["dict", [[snapshot(key), snapshot(item)] for key, item in value.items()]]
        if isinstance(value, BaseType) and compiler_owned(type(value)):
            return ["type", type(value).__module__, type(value).__qualname__, snapshot(vars(value))]
        if isinstance(value, InstructionGroup) and compiler_owned(type(value)):
            return ["instruction_group", snapshot(type(value)), snapshot(vars(value))]
        if isinstance(value, (staticmethod, classmethod)):
            return [type(value).__name__, snapshot(value.__func__)]
        if isinstance(value, types.FunctionType):
            return function(value)
        if isinstance(value, type):
            if compiler_owned(value):
                # Source fingerprints cover compiler implementations, but public
                # class defaults can be configured by user code at runtime.
                defaults = {}
                for base in reversed(value.__mro__):
                    for name, member in vars(base).items():
                        if (
                            name.startswith("_")
                            or callable(member)
                            or isinstance(member, (staticmethod, classmethod, property))
                        ):
                            defaults.pop(name, None)
                        else:
                            defaults[name] = member
                return ["compiler_type", value.__module__, value.__qualname__, snapshot(defaults)]
            if type(value) is not type:
                raise _UntrackedDependency
            members = []
            for name, member in sorted(vars(value).items()):
                if name in ("__dict__", "__weakref__", "__module__", "__doc__"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    members.append([name, type(member).__name__, snapshot(member.__func__)])
                elif isinstance(member, property):
                    members.append([name, "property", snapshot((member.fget, member.fset, member.fdel))])
                else:
                    members.append([name, "value", snapshot(member)])
            return [
                "class",
                value.__module__,
                value.__qualname__,
                inspect.getsource(value),
                [snapshot(base) for base in value.__bases__],
                members,
            ]
        raise _UntrackedDependency

    def function(func: types.FunctionType) -> Any:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
        definition = tree.body[0]
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # These decorators have already been applied; snapshot the resulting
            # function and descriptor instead of evaluating decorator globals.
            definition.decorator_list = []
        # String annotations are evaluated against the method's external env by
        # the frontend, even though their names are absent from Python bytecode.
        for annotation in func.__annotations__.values():
            if isinstance(annotation, str):
                tree.body.append(ast.Expr(value=ast.parse(annotation, mode="eval").body))
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
            raise _UntrackedDependency
        # Reflection can introduce dependencies that static names do not expose.
        if any(
            isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__"
            for node in ast.walk(tree)
        ):
            raise _UntrackedDependency
        env = func.__globals__.copy()
        cells = func.__closure__ or ()
        closure = dict(zip(func.__code__.co_freevars, (cell.cell_contents for cell in cells)))
        env.update(closure)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        dependencies = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            name = node.id
            if name in env:
                value = env[name]
            elif name in func.__builtins__:
                value = func.__builtins__[name]
            else:
                # Nested-function parameters and other lexical names are already
                # represented by the enclosing source/code.
                continue
            key = name
            current = node
            while True:
                parent = parents.get(current)
                attribute_access = isinstance(parent, ast.Attribute) and parent.value is current
                if isinstance(value, types.ModuleType):
                    if not attribute_access or parent.attr not in vars(value):
                        raise _UntrackedDependency
                    dependencies[key] = ["module", value.__name__]
                    member = vars(value)[parent.attr]
                elif isinstance(value, type) and attribute_access:
                    dependencies[key] = value
                    try:
                        member = inspect.getattr_static(value, parent.attr)
                    except AttributeError:
                        raise _UntrackedDependency from None
                else:
                    break
                key += "." + parent.attr
                value = member
                current = parent
            if value is object and isinstance(parents.get(current), ast.Call):
                raise _UntrackedDependency
            dependencies[key] = value
        return [
            "function",
            source,
            marshal.dumps(func.__code__).hex(),
            snapshot(func.__defaults__),
            snapshot(func.__kwdefaults__),
            snapshot(func.__annotations__),
            snapshot(vars(func)),
            [[name, snapshot(value)] for name, value in sorted(closure.items())],
            [[name, snapshot(value)] for name, value in sorted(dependencies.items())],
        ]

    try:
        payload = [sys.version, np.__version__, snapshot(Attributes), snapshot(script_cls), snapshot(inputs)]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    except (_UntrackedDependency, OSError, TypeError, ValueError, SyntaxError, RecursionError):
        return None
