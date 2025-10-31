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
import re
from enum import Enum, auto
from html import escape
from typing import Any, Generator, List, Tuple

from jinja2 import BaseLoader, Environment


class TokenKind(Enum):
    COMMENT = auto()
    KEYWORD = auto()
    VARIABLE = auto()
    IDENTIFIER = auto()
    FLOAT_LITERAL = auto()
    INTEGER_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    STRING = auto()
    PUNCTUATION = auto()
    DATATYPE = auto()
    BUILTIN = auto()
    INSTRUCTION = auto()
    OPERATOR = auto()
    WHITESPACE = auto()
    UNKNOWN = auto()


# RGB color map for each token kind
COLOR_MAP = {
    TokenKind.COMMENT: (128, 128, 128),
    TokenKind.KEYWORD: (0, 0, 255),
    TokenKind.VARIABLE: (32 - 16, 64 + 32 - 16, 96 + 32 - 16),
    TokenKind.IDENTIFIER: (0, 0, 0),
    TokenKind.FLOAT_LITERAL: (139, 0, 0),
    TokenKind.INTEGER_LITERAL: (139, 0, 0),
    TokenKind.BOOLEAN_LITERAL: (139, 0, 0),
    TokenKind.STRING: (0, 128, 0),
    TokenKind.DATATYPE: (0, 102, 51),
    TokenKind.BUILTIN: (0, 102, 51),
    TokenKind.INSTRUCTION: (0, 0, 0),
    TokenKind.PUNCTUATION: (0, 0, 0),
    TokenKind.OPERATOR: (0, 102, 204),
    TokenKind.WHITESPACE: None,
    TokenKind.UNKNOWN: (255, 0, 0),
}


def instruction_names() -> Generator[Any, Any, None]:
    import tilus.ir.instructions
    from tilus.ir.instructions import Instruction

    for name in dir(tilus.ir.instructions):
        obj = getattr(tilus.ir.instructions, name)
        if isinstance(obj, type) and issubclass(obj, Instruction) and obj is not Instruction:
            yield obj.__name__.removesuffix("Inst")


instruction_pattern = r"\b(" + "|".join(re.escape(instr) for instr in instruction_names()) + r")\b"

TOKEN_RULES: List[Tuple[str, TokenKind]] = [
    (r"#.*", TokenKind.COMMENT),
    (r"\blet\b|\bdef\b|\bdeclare\b|\bfor\b|\bin\b|\baddr\b", TokenKind.KEYWORD),
    (r"%[a-zA-Z_][a-zA-Z0-9_]*", TokenKind.VARIABLE),
    (r"\d+\.\d+f", TokenKind.FLOAT_LITERAL),
    (r"\d+", TokenKind.INTEGER_LITERAL),
    (r"\bTrue\b|\bFalse\b", TokenKind.BOOLEAN_LITERAL),
    (r'"[^"\n]*"|\'[^\'\n]*\'', TokenKind.STRING),
    (r"\b(float16|float32|int32|int64|bool)\b", TokenKind.DATATYPE),
    (instruction_pattern, TokenKind.INSTRUCTION),
    (r"(<=|>=|!=|\^|%|==|&&|\|\||[+\-*/~<>])", TokenKind.OPERATOR),
    (r"[=:\[\]\(\)\{\},*\.]", TokenKind.PUNCTUATION),
    (r"\b[A-Za-z_][A-Za-z0-9_]*\b", TokenKind.IDENTIFIER),
    (r"\s+", TokenKind.WHITESPACE),
]


def tokenize(text: str) -> List[Tuple[TokenKind, str]]:
    idx = 0
    tokens = []
    while idx < len(text):
        for pattern, kind in TOKEN_RULES:
            regex = re.compile(pattern)
            match = regex.match(text, idx)
            if match:
                value = match.group(0)
                tokens.append((kind, value))
                idx += len(value)
                break
        else:
            tokens.append((TokenKind.UNKNOWN, text[idx]))
            idx += 1
    return tokens


def split_into_lines(tokens: List[Tuple[TokenKind, str]]) -> List[List[Tuple[TokenKind, str]]]:
    lines = []
    current: list[tuple[TokenKind, str]] = []
    for kind, text in tokens:
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append(current)
                current = []
            if part or kind == TokenKind.WHITESPACE:
                current.append((kind, part))
    if current:
        lines.append(current)
    return lines


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>DSL Program Highlights</title>
  <style>
    body {
      font-family: monospace;
      margin: 0;
      display: flex;
    }
    nav {
      position: fixed;
      top: 0;
      left: 0;
      width: 200px;
      height: 100vh;
      background: #f0f0f0;
      border-right: 1px solid #ccc;
      overflow-y: auto;
      padding: 1em;
    }

    .nav-list {
      display: flex;
      flex-direction: column;
      gap: 0.5em; /* space between links, without breaking hitboxes */
    }

    nav a {
      display: block;
      padding: 0.25em 0.5em;
      border-radius: 4px;
      color: #333;
      text-decoration: none;
      transition: background-color 0.2s ease, color 0.2s ease;
    }

    nav a:hover {
      background-color: #e0e0e0;
      color: #000;
      text-decoration: none;
    }
    main {
      padding: 2em;
      margin-left: 200px;
      padding-left: 2em;
      flex-grow: 1;
      overflow-x: auto;
    }
    pre {
      background: #fff;
      padding: 1em;
      border: 1px solid #ccc;
      line-height: 1.4;
      white-space: pre-wrap;
    }
    h2 {
      # border-bottom: 1px solid #ccc;
      # padding-bottom: 0.25em;
      padding: 1.00em 0.25em 0.25em 0.5em; /* top, right, bottom, left */
      # margin-left: -0.5em; /* align border with rest of text */
    }
  </style>
</head>
<body>
<nav>
  <strong>Programs</strong>
  {% for name in programs.keys() %}
    <a href="#{{ name }}">{{ name }}</a>
  {% endfor %}
</nav>
<main>
  {% for name, lines in rendered.items() %}
    <section id="{{ name }}">
      <h2>{{ name }}</h2>
      <pre>{{ lines | join('\n') }}</pre>
    </section>
  {% endfor %}
</main>
</body>
</html>
"""


def highlight(programs: dict[str, str]) -> str:
    def render_line(line):
        parts = []
        for kind, text in line:
            color = COLOR_MAP.get(kind)
            escaped = escape(text)
            if color:
                parts.append(f'<span style="color: rgb({color[0]}, {color[1]}, {color[2]})">{escaped}</span>')
            else:
                parts.append(escaped)
        return "".join(parts)

    rendered = {}
    for name, text in programs.items():
        tokens = tokenize(text)
        lines = split_into_lines(tokens)
        rendered[name] = [render_line(line) for line in lines]

    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(TEMPLATE)
    return template.render(programs=programs, rendered=rendered)


# Example usage
if __name__ == "__main__":
    programs = {
        "matmul_v0": """def matmul_v0(
    %v0: int32,
    %p0: float16*,
    %p1: float16*,
    %p2: float16*
):
    # initialize
    let %v1: int32 = (16 * blockIdx.x)
""",
        "matmul_v1": """def matmul_v1(
    %v0: int32,
    %p0: float32*,
    %p1: float32*,
    %p2: float32*
):
    let %v1: int32 = (blockIdx.y * 64)
    # inner loop
    for %v2 in range(8):
        let %v3: int32 = (%v2 * 4)
""",
    }
    html = highlight(programs)
    with open("highlighted_programs.html", "w", encoding="utf-8") as f:
        f.write(html)
