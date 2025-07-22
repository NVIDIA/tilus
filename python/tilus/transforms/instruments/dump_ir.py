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
import shutil
import time
from pathlib import Path

import tabulate

from tilus.ir.prog import Program
from tilus.ir.tools import IRPrinter
from tilus.transforms.instruments.instrument import PassInstrument
from tilus.transforms.instruments.utils.highlight import highlight


class DumpIRInstrument(PassInstrument):
    def __init__(self, dump_dir: Path):
        self.dump_dir: Path = dump_dir
        self.count: int = 0
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.start_time: dict[str, float] = {}
        self.elapsed_time: dict[str, float] = {}
        self.programs: dict[str, str] = {}

    def before_all_passes(self, program: Program) -> None:
        printer = IRPrinter()
        # remove the old dump directory
        shutil.rmtree(self.dump_dir, ignore_errors=True)

        self.dump_dir.mkdir(parents=True, exist_ok=True)
        program_text = str(printer(program))
        with open(self.dump_dir / "0_Original.txt", "w") as f:
            f.write(program_text)

        self.programs["0. Original"] = program_text

        self.count = 1

    def before_pass(self, pass_name: str, program: Program) -> None:
        self.start_time[pass_name] = time.time()

    def after_pass(self, pass_name: str, program: Program) -> None:
        self.elapsed_time[pass_name] = time.time() - self.start_time[pass_name]

        printer = IRPrinter()
        file_name = f"{self.count}_{pass_name}"
        program_text = str(printer(program))
        with open(self.dump_dir / f"{file_name}.txt", "w") as f:
            f.write(program_text)
        self.programs[f"{self.count}. {pass_name}"] = program_text

        self.count += 1

    def after_all_passes(self, program: Program) -> None:
        # output the lower time for each pass
        headers = ["Pass", "Time"]
        rows = []
        for name, elapsed_time in self.elapsed_time.items():
            rows.append([name, "{:.3f} seconds".format(elapsed_time)])
        with open(self.dump_dir / "lower_time.txt", "w") as f:
            f.write(tabulate.tabulate(rows, headers=headers))

        # output the rendered programs to HTML
        with open(self.dump_dir / "programs.html", "w", encoding="utf-8") as f:
            f.write(highlight(self.programs))
