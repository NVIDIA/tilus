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
from typing import Sequence

from tilus.hidet.ir.module import IRModule

from .add_explicit_cast import add_explicit_cast_pass
from .add_hints import add_hints_pass
from .annotate_header_and_libs import annotate_header_and_libs_pass
from .base import FunctionPass, Pass, PassContext, RepeatFunctionPass, SequencePass
from .bind_predefined_variables import bind_predefined_variables_pass
from .check_launch_configuration import check_launch_configuration_pass

# Extension passes
from .deadcode_elimination import deadcode_elimination_pass as lowlevel_deadcode_elimination_pass
from .declare_to_let import declare_to_let_pass
from .expand_let_expr import expand_let_expr_pass
from .explicit_unroll import explicit_unroll_pass
from .flatten_tensor_index import flatten_tensor_index_pass
from .flatten_tensor_slice import flatten_tensor_slice_pass
from .generate_launch_func import generate_launch_func_pass
from .hoist_loop_invariants import hoist_loop_invariants_pass
from .import_primitive_functions import import_primitive_functions_pass
from .inline_function import inline_function_pass
from .inline_let_stmt import inline_let_stmt_pass
from .instantiate_symbols import instantiate_symbols_pass
from .instruments import PassInstrument, ProfileInstrument, SaveIRInstrument
from .lower_affine_to_recurence import lower_affine_to_recurrence_pass
from .lower_integer_subbyte import lower_integer_subbyte_pass
from .lower_special_cast import lower_special_cast_pass
from .lower_subbyte_type import lower_subbyte_type_pass
from .lower_task_mapping import lower_task_mapping_pass
from .propagate_launch_bound import propagate_launch_bound_pass
from .resolve_generic_primitive_function import resolve_primitive_func_pass
from .rule_based_simplifier import rule_based_simplify_pass
from .simplify_addition_chain import simplify_addition_chain_pass
from .simplify_stmt import simplify_stmt_pass

# Passes used by tilus/drivers.py
from .unify_global_objects import unify_global_objects_pass


def lower_with(ir_module: IRModule, transforms: Sequence[Pass]) -> IRModule:
    """Apply a sequence of transforms to an IR module."""
    ctx = PassContext.current()
    for instrument in ctx.instruments:
        instrument.before_all_passes(ir_module)
    for transform in transforms:
        ir_module = transform(ir_module)
    for instrument in ctx.instruments:
        instrument.after_all_passes(ir_module)
    return ir_module
