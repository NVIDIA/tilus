# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from hidet.runtime.compiled_module import CompiledModule, compiled_module_exists


class CompiledProgram:
    def __init__(self, program_dir: str | Path):
        self.program_dir: Path = Path(program_dir)
        self.compiled_module = CompiledModule(str(self.program_dir / "module"))

    def __call__(self, *args):
        return self.compiled_module(*args)


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
        [compiled_module_exists(str(path / "module")), (path / "program.txt").exists(), (path / "options.txt").exists()]
    )
