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
from pathlib import Path

import tvm_ffi


class CompiledModule:
    """A compiled module loaded from a shared library, callable via the 'launch' function."""

    def __init__(self, lib_path: str | Path):
        self.lib_path: Path = Path(lib_path)
        self.module = tvm_ffi.load_module(str(self.lib_path))
        self.launch_func = self.module["launch"]

    def __call__(self, *args):
        return self.launch_func(*args)

    def __getitem__(self, name):
        return self.module[name]


class CompiledProgram:
    def __init__(self, program_dir: str | Path):
        self.program_dir: Path = Path(program_dir)
        self.compiled_module = tvm_ffi.load_module(str(Path(self.program_dir) / "module" / "lib.so"))
        self.launch_func = self.compiled_module["launch"]

    def get_launch_func(self) -> tvm_ffi.Function:
        return self.launch_func

    def __call__(self, *args):
        return self.launch_func(*args)


def load_compiled_program(program_dir: str | Path) -> CompiledProgram:
    """
    Load a compiled program from the cache directory.

    Parameters
    ----------
    program_dir: str or Path
        The cache directory of the compiled program.

    Returns
    -------
    compiled_program: CompiledProgram
        The compiled program.
    """
    return CompiledProgram(program_dir)


def compiled_program_exists(cache_dir: Path | str) -> bool:
    """
    Check if there is a program that has been built and cached under the given program cache dir.

    Parameters
    ----------
    cache_dir: Path | str
        The cache directory of the compiled program.

    Returns
    -------
    ret: bool
        True if the program exists, False otherwise.
    """
    path = Path(cache_dir)
    return all(
        [(path / "module" / "lib.so").exists(), (path / "program.txt").exists(), (path / "options.txt").exists()]
    )
