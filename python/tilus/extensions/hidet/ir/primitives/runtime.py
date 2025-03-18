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
from typing import Union

from hidet.ir.expr import Expr
from hidet.ir.type import FuncType, void_p, void, string_type
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from tilus.extensions.hidet.ir.expr import as_expr
from hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function(
        name="set_symbol_value_ptr",
        func_or_type=FuncType([string_type(), void_p], void),
        codegen_name="set_symbol_value_ptr",
    )
    register_primitive_function(
        name="get_symbol_value_ptr", func_or_type=FuncType([string_type()], void_p), codegen_name="get_symbol_value_ptr"
    )


def get_symbol_value_ptr(name: Union[str, Expr]) -> Expr:
    return call_primitive_func("get_symbol_value_ptr", [as_expr(name)])


def set_symbol_value_ptr(name: Union[str, Expr], value: Expr) -> Expr:
    return call_primitive_func("set_symbol_value_ptr", [as_expr(name), value])
