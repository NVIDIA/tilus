from typing import Sequence

from .base import Pass, PassContext, apply_transforms
from .bound_aware_simplify import bound_aware_simplify_pass
from .declare_to_let import declare_to_let_pass
from .inject_print_instruction import inject_print_instruction_pass
from .lower_load_store import lower_load_store_pass
from .lower_param_only_expr import lower_param_only_expr_pass
from .lower_to_load_matrix import lower_to_load_matrix_pass
from .scalar_analyze import analyze_scalar_pass


def get_default_passes() -> list[Pass]:
    return [
        declare_to_let_pass(),
        lower_param_only_expr_pass(),
        lower_to_load_matrix_pass(),
        lower_load_store_pass(),
        bound_aware_simplify_pass(),
        analyze_scalar_pass(),
    ]
