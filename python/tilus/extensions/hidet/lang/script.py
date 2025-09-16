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
from __future__ import annotations

import ast as py_ast
import inspect
from types import FunctionType
from typing import Tuple, List, Any, Dict

from hidet.ir.func import Function
from tilus.extensions.hidet.lang.transpiler import ExtendedPythonToHidetTranslator


def eliminate_indent(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    indent = len(source)
    for line in lines:
        if len(line.strip()) == 0:
            continue
        indent = min(indent, len(line) - len(line.lstrip()))
    source = '\n'.join([line[indent:] for line in lines])
    return source, indent


def eliminate_decorators(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    num_decorators = 0
    for line in lines:
        if len(line) > 0 and line[0] == '@':
            num_decorators += 1
        else:
            break
    source = '\n'.join(lines[num_decorators:])
    return source, num_decorators


def script(func: FunctionType) -> Function:
    """
    Decorator to convert a Python function to a Hidet function.

    Parameters
    ----------
    func: FunctionType
        The python function to be converted to a Hidet function.

    Returns
    -------
    ret: Function
        The hidet.ir.Function that is converted from the given Python function.
    """
    # Extract the source code of given function
    lines, start_line = inspect.getsourcelines(func)
    file = inspect.getsourcefile(func)
    source = ''.join(lines)
    source, col_offset = eliminate_indent(source)
    source, inc_lineno = eliminate_decorators(source)
    start_line += inc_lineno
    parsed: py_ast.AST = py_ast.parse(source=source)

    # Get the environment (globals and binding of free variables)
    # See the data model of python for the details of func.__globals__, func.__closure__ and func.__code__:
    #     https://docs.python.org/3/reference/datamodel.html
    env: Dict[str, Any] = func.__globals__.copy()
    func_freevar_names: List[str] = list(func.__code__.co_freevars)
    func_freevar_cells: List[Any] = [v.cell_contents for v in func.__closure__] if func.__closure__ else []
    assert len(func_freevar_names) == len(func_freevar_cells)
    env.update(dict(zip(func_freevar_names, func_freevar_cells)))

    # get the type annotations of function parameters.
    func_annotations: Dict[str, Any] = func.__annotations__

    # Translate the Python function into Hidet function
    translator = ExtendedPythonToHidetTranslator(
        file=file, start_lineno=start_line, start_column=col_offset, env=env, func_annotations=func_annotations
    )
    hidet_function = translator(parsed)

    assert isinstance(hidet_function, Function)
    return hidet_function
