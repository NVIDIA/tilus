# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal option registry, stripped from hidet v0.6.1 option.py."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional


class OptionRegistry:
    """Registry for option definitions."""

    registered_options: Dict[str, OptionRegistry] = {}

    def __init__(
        self,
        name: str,
        type_hint: str,
        description: str,
        default_value: Any,
        normalizer: Optional[Callable[[Any], Any]] = None,
        choices: Optional[Iterable[Any]] = None,
        checker: Optional[Callable[[Any], bool]] = None,
        env: Optional[str] = None,
    ):
        self.name = name
        self.type_hint = type_hint
        self.description = description
        self.default_value = default_value
        self.normalizer = normalizer
        self.choices = choices
        self.checker = checker
        self.env = env


def register_option(
    name: str,
    type_hint: str,
    description: str,
    default_value: Any,
    normalizer: Optional[Callable[[Any], Any]] = None,
    choices: Optional[Iterable[Any]] = None,
    checker: Optional[Callable[[Any], bool]] = None,
    env: Optional[str] = None,
):
    """Register an option."""
    registered_options = OptionRegistry.registered_options
    if name in registered_options:
        raise KeyError(f"Option {name} has already been registered.")
    registered_options[name] = OptionRegistry(
        name, type_hint, description, default_value, normalizer, choices, checker, env
    )


class OptionContext:
    """The option context."""

    stack: List[OptionContext] = []

    def __init__(self):
        self.options: Dict[str, Any] = {}

    def __enter__(self):
        OptionContext.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        OptionContext.stack.pop()

    @staticmethod
    def current() -> OptionContext:
        return OptionContext.stack[-1]

    def set_option(self, name: str, value: Any):
        if name not in OptionRegistry.registered_options:
            raise KeyError(f"Option {name} has not been registered.")
        registry = OptionRegistry.registered_options[name]
        if registry.normalizer is not None:
            value = registry.normalizer(value)
        if registry.checker is not None:
            if not registry.checker(value):
                raise ValueError(f"Invalid value for option {name}: {value}")
        if registry.choices is not None:
            if value not in registry.choices:
                raise ValueError(f"Invalid value for option {name}: {value}, choices {registry.choices}")
        self.options[name] = value

    def get_option(self, name: str) -> Any:
        for ctx in reversed(OptionContext.stack):
            if name in ctx.options:
                return ctx.options[name]
        if name not in OptionRegistry.registered_options:
            raise KeyError(f"Option {name} has not been registered.")
        registry = OptionRegistry.registered_options[name]
        return registry.default_value


# Initialize the default context
OptionContext.stack.append(OptionContext())


def set_option(name: str, value: Any):
    """Set the value of an option in current option context."""
    OptionContext.current().set_option(name, value)


def get_option(name: str) -> Any:
    """Get the value of an option in current option context."""
    return OptionContext.current().get_option(name)


def context() -> OptionContext:
    """Create a new option context."""
    return OptionContext()


def current_context() -> OptionContext:
    """Get the current option context."""
    return OptionContext.current()


def dump_options() -> Dict[str, Any]:
    """Dump the options in option context stack."""
    return {"option_context_stack": OptionContext.stack, "registered_options": OptionRegistry.registered_options}


def restore_options(dumped_options: Dict[str, Any]):
    """Restore the options from dumped options."""
    OptionContext.stack = dumped_options["option_context_stack"]
    OptionRegistry.registered_options = dumped_options["registered_options"]
