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
from __future__ import annotations

import ast
import os
from typing import Optional

from hidet.lang.transpiler import PythonAstFunctor
from hidet.utils import str_indent


class TilusProgramError(Exception):
    def __init__(self, translator: PythonAstFunctor, obj: Optional[ast.stmt | ast.expr], msg: str):
        super().__init__(translator, obj, msg)  # make this exception picklable
        self.file: str = translator.file
        self.start_lineno: int = translator.start_lineno
        self.start_column: int = translator.start_column
        self.obj: Optional[ast.stmt | ast.expr] = obj
        self.msg: str = msg

    def __str__(self):
        assert self.obj is not None
        lineno = self.start_lineno + self.obj.lineno
        column = self.start_column + self.obj.col_offset
        lines = []
        if not os.path.exists(self.file):
            source_line = ""
        else:
            with open(self.file, "r") as f:
                source_lines = list(f.readlines())
                if lineno < len(source_lines):
                    source_line = source_lines[lineno - 2].rstrip()
                else:
                    source_line = ""
        lines.append("")
        lines.append(
            "  File {file}:{line}:{column}:".format(file=os.path.abspath(self.file), line=lineno - 1, column=column)
        )
        if source_line:
            lines.append(source_line)
            lines.append(" " * column + "^")
        if source_line and "\n" not in self.msg:
            indent = column
        else:
            indent = 4
        lines.append("{msg}".format(msg=str_indent(self.msg, indent=indent)))
        return "\n".join(lines)
