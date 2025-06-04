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
from typing import List, Sequence

from hidet.ir.cute.collective import CollectiveStore
from hidet.ir.module import IRModule
from hidet.transforms.add_explicit_cast import add_explicit_cast_pass
from hidet.transforms.add_hints import add_hints_pass
from hidet.transforms.annotate_header_and_libs import annotate_header_and_libs_pass
from hidet.transforms.attach_hash_to_signature import attach_hash_to_signature
from hidet.transforms.base import FunctionPass, Pass, PassContext, RepeatFunctionPass, SequencePass
from hidet.transforms.check_launch_configuration import check_launch_configuration_pass
from hidet.transforms.convert_div_to_fastintdiv import convert_div_to_fastintdiv_pass
from hidet.transforms.cute.cuda.instantiate_auto_annotation import instantiate_auto_annotation_pass
from hidet.transforms.cute.cuda.instruction_selection import instruction_selection_pass
from hidet.transforms.cute.cuda.lower_cute_dialect import lower_cute_dialect_pass
from hidet.transforms.cute.cuda.resolve_bank_conflict import resolve_bank_conflict_pass
from hidet.transforms.cute.cuda.shared_memory_allocation import shared_memory_allocation_pass
from hidet.transforms.cute.cuda.vectorize_elementwise import vectorize_elementwise_pass
from hidet.transforms.cute.generic.canonicalize import canonicalize_pass
from hidet.transforms.cute.generic.canonicalize_arithmetic_expression import canonicalize_arithmetic_expression_pass
from hidet.transforms.cute.generic.deadcode_elimination import deadcode_elimination_pass
from hidet.transforms.declare_to_let import declare_to_let_pass
from hidet.transforms.expand_let_expr import expand_let_expr_pass
from hidet.transforms.expand_repeat import expand_repeat_mapping_pass
from hidet.transforms.explicit_unroll import explicit_unroll_pass
from hidet.transforms.flatten_tensor_index import flatten_tensor_index_pass
from hidet.transforms.flatten_tensor_slice import flatten_tensor_slice_pass
from hidet.transforms.generate_launch_func import generate_launch_func_pass
from hidet.transforms.import_primitive_functions import import_primitive_functions_pass
from hidet.transforms.inline_function import inline_function_pass
from hidet.transforms.inline_let_stmt import inline_let_stmt_pass
from hidet.transforms.instantiate_symbols import instantiate_symbols_pass
from hidet.transforms.instruments import PassInstrument, ProfileInstrument, SaveIRInstrument
from hidet.transforms.lower_integer_subbyte import lower_integer_subbyte_pass
from hidet.transforms.lower_protect_access import lower_protect_access_pass
from hidet.transforms.lower_special_cast import lower_special_cast_pass
from hidet.transforms.lower_task_mapping import lower_task_mapping_pass
from hidet.transforms.normalize_const_tensor import normalize_const_tensor_pass
from hidet.transforms.propagate_launch_bound import propagate_launch_bound_pass
from hidet.transforms.resolve_generic_primitive_function import resolve_primitive_func_pass
from hidet.transforms.rule_based_simplifier import rule_based_simplify_pass
from hidet.transforms.simplify_addition_chain import simplify_addition_chain_pass
from hidet.transforms.simplify_stmt import simplify_stmt_pass
from hidet.transforms.spatial_simplification import spatial_simplification_pass
from hidet.transforms.task_mapping_bound_check import task_mapping_bound_check
from hidet.transforms.unify_global_objects import unify_global_objects_pass

from tilus.extensions.hidet.transforms.deadcode_elimination import (
    deadcode_elimination_pass as lowlevel_deadcode_elimination_pass,
)

# predefined pass groups
cute_generic_transforms: List[Pass] = [
    canonicalize_arithmetic_expression_pass(),
    canonicalize_pass(),
    deadcode_elimination_pass(),
]

cute_cuda_transforms: List[Pass] = [
    lower_cute_dialect_pass((CollectiveStore,)),
    instantiate_auto_annotation_pass(),
    vectorize_elementwise_pass(),
    instruction_selection_pass(),
    resolve_bank_conflict_pass(),
    instruction_selection_pass(),
    shared_memory_allocation_pass(),
    lower_cute_dialect_pass(),
]

common_transforms = [
    # necessary passes
    attach_hash_to_signature(),
    unify_global_objects_pass(),
    generate_launch_func_pass(),
    propagate_launch_bound_pass(),
    flatten_tensor_slice_pass(),
    lower_protect_access_pass(),
    spatial_simplification_pass(),
    flatten_tensor_index_pass(),
    task_mapping_bound_check(),  # this pass assume that propagate_launch_bound_pass() will be run before
    expand_repeat_mapping_pass(),
    lower_task_mapping_pass(),
    normalize_const_tensor_pass(),
    declare_to_let_pass(),
    rule_based_simplify_pass(),  # make ir more readable
    flatten_tensor_index_pass(),
    lower_special_cast_pass(),
    inline_function_pass(),
    resolve_primitive_func_pass(),
    import_primitive_functions_pass(),
    resolve_primitive_func_pass(),
    import_primitive_functions_pass(),
    lower_integer_subbyte_pass(),
    add_explicit_cast_pass(),
    declare_to_let_pass(),
    instantiate_symbols_pass(),
    convert_div_to_fastintdiv_pass(),
    import_primitive_functions_pass(),
    check_launch_configuration_pass(),
    # simplification
    expand_let_expr_pass(),
    inline_let_stmt_pass(),
    explicit_unroll_pass(),
    rule_based_simplify_pass(),
    simplify_addition_chain_pass(),
    add_hints_pass(),
    inline_let_stmt_pass(),
    simplify_stmt_pass(),
    lowlevel_deadcode_elimination_pass(),
    annotate_header_and_libs_pass(),
]


def lower_with(ir_module: IRModule, transforms: Sequence[Pass]) -> IRModule:
    ctx = PassContext.current()
    for instrument in ctx.instruments:
        instrument.before_all_passes(ir_module)
    for transform in transforms:
        ir_module = transform(ir_module)
    for instrument in ctx.instruments:
        instrument.after_all_passes(ir_module)

    return ir_module


def lower(ir_module: IRModule) -> IRModule:
    ir_module = lower_with(ir_module, cute_generic_transforms + cute_cuda_transforms + common_transforms)
    return ir_module
