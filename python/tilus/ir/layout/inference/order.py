from typing import Type

from tilus.ir.layout.inference.rule import LayoutInferenceRule
from tilus.utils import initialize

from .inference_rules.assign import AssignRule
from .inference_rules.cp_async import CopyAsyncRule
from .inference_rules.elementwise_binary import BinaryRule
from .inference_rules.elementwise_unary import UnaryRule
from .inference_rules.empty_rule import EmptyRule
from .inference_rules.load_global import LoadGlobalRule
from .inference_rules.load_shared import LoadSharedInferRegisterRule, LoadSharedInferSwizzledSharedRule
from .inference_rules.mma_dot import MmaDotRule
from .inference_rules.reduce import ReduceRule
from .inference_rules.shared_slice import SharedSliceRule
from .inference_rules.store_shared import StoreSharedSwizzleRule
from .inference_rules.transpose import TransposeRule
from .inference_rules.where import WhereRule

inference_order: list[list[Type[LayoutInferenceRule]]] = [
    [MmaDotRule],
    [BinaryRule, UnaryRule],
    [LoadGlobalRule],
    [ReduceRule],
    [TransposeRule],
    [WhereRule],
    [AssignRule],
    [EmptyRule],
    # shared memory rules
    [LoadSharedInferSwizzledSharedRule, StoreSharedSwizzleRule],
    [SharedSliceRule],
    [CopyAsyncRule],
    [LoadSharedInferRegisterRule],
]

rule2order: dict[Type[LayoutInferenceRule], int] = {}


@initialize()
def init_rule_sort_key() -> None:
    count = 0
    for rule_group in inference_order:
        for rule in rule_group:
            rule2order[rule] = count
            count += 1
