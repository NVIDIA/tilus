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
# pylint: disable=subprocess-run-check
import argparse
import importlib
import inspect
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any


def resolve_binary(name: str) -> str:
    """Resolve the path to a binary by checking PATH, then common install locations."""
    path = shutil.which(name)
    if path:
        return path
    for candidate in [f"/usr/local/bin/{name}", f"/usr/local/cuda/bin/{name}"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(f"Could not find '{name}' executable.")


class ProfilerReport:
    """Report from a profiling run that can be visualized in the corresponding UI."""

    def __init__(self, report_path: str, ui_binary: str):
        self.report_path: str = report_path
        self._ui_binary = ui_binary

    def visualize(self):
        ui_path = resolve_binary(self._ui_binary)
        subprocess.run(f"{ui_path} {self.report_path}", shell=True)


class Profiler:
    """Base class for NVIDIA profiler wrappers (ncu, nsys).

    Handles binary resolution, report path management, argument pickling,
    and subprocess invocation with a tool-specific command template.

    The command template should use these placeholders:
        {profiler_path}, {report_path}, {python_executable}, {python_script}, {args}
    Plus any tool-specific placeholders passed via extra_template_kwargs in run().
    """

    def __init__(
        self,
        binary_name: str,
        ui_binary_name: str,
        command_template: str,
        report_dir: str,
        report_ext: str,
        display_name: str,
        entry_script: str,
    ):
        self.binary_name = binary_name
        self.ui_binary_name = ui_binary_name
        self.command_template = command_template
        self.report_dir = report_dir
        self.report_ext = report_ext
        self.display_name = display_name
        self.entry_script = entry_script

    def run(
        self,
        func: Any,
        func_args: tuple = (),
        func_kwargs: dict | None = None,
        extra_template_kwargs: dict | None = None,
    ) -> ProfilerReport:
        func_kwargs = func_kwargs or {}

        script_path: str = inspect.getfile(func)
        func_name: str = func.__name__

        # find an unused report path
        report_template = os.path.join(os.path.dirname(script_path), f"{self.report_dir}/report{{}}.{self.report_ext}")
        idx = 0
        while os.path.exists(report_template.format(idx)):
            idx += 1
        report_path = report_template.format(idx)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # pickle function arguments
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            args_path = f.name
            pickle.dump((func_args, func_kwargs), f)

        template_kwargs = {
            "profiler_path": resolve_binary(self.binary_name),
            "report_path": report_path,
            "python_executable": sys.executable,
            "python_script": self.entry_script,
            "args": f"{script_path} {func_name} {args_path}",
        }
        if extra_template_kwargs:
            template_kwargs.update(extra_template_kwargs)

        command = self.command_template.format(**template_kwargs)
        print(f"Running {self.display_name} command:")
        print(command.replace("--", "\n\t--"))

        status = subprocess.run(command, shell=True)
        if status.returncode != 0:
            raise RuntimeError(f"Error when running {self.display_name}.")

        return ProfilerReport(report_path, self.ui_binary_name)


def run_profiled_func(script_path: str, func_name: str, args_pickled_path: str) -> None:
    """Entry point for the subprocess: import the target module and call the function."""
    with open(args_pickled_path, "rb") as f:
        args, kwargs = pickle.load(f)

    # remove the dir path of the current script from sys.path to avoid module overriding
    sys.path = [path for path in sys.path if not path.startswith(os.path.dirname(__file__))]

    try:
        sys.path.append(os.path.dirname(script_path))
        module = importlib.import_module(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError(f"Can not import the python script: {script_path}") from e

    if not hasattr(module, func_name):
        raise RuntimeError(f'Can not find the function "{func_name}" in {script_path}')

    func = getattr(module, func_name)

    try:
        func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f'Error when running the function "{func_name}"') from e


def profiler_main():
    """Shared main() for profiler entry-point scripts."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("script_path", type=str)
        parser.add_argument("func", type=str)
        parser.add_argument("args", type=str)
        args = parser.parse_args()
        run_profiled_func(args.script_path, args.func, args.args)
    except Exception as e:
        print(f"Error when running the script: {e}")
        print("Traceback:")
        traceback.print_exc()
        raise
