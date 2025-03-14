from typing import Dict, Optional, Union, List, Tuple
from hidet.ir.dtypes import int32
from hidet.ir.expr import Var, Expr
from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier as OriginalRuleBasedSimplifier
from hidet.transforms.rule_based_simplifier import BoundInfo, BoundAnalyzer
from hidet.utils import repeat_until_converge


class RuleBasedSimplifier(OriginalRuleBasedSimplifier):
    def __init__(self, var2bound: Optional[Dict[Var, BoundInfo]]):
        super().__init__()
        self.analyzer = BoundAnalyzer(var2bound)  # type: ignore

        e1, e2, c1, c2, ec1, ec2 = self.args

        extra_patterns = [
            ((e1 + c1) - c2, e1 + (c1 - c2), c1 >= c2),
            ((e1 + c1) - c2, e1 - (c2 - c1), c1 < c2),
            ((e1 + c1) // c2, e1 // c2 + c1 // c2, c1 % c2 == 0),
            ((e1 % c1), int32.zero, c1 == 1),
            ((e1 % c1) % c2, e1 % c2, c1 % c2 == 0),
            (e1 / c1 % c2, e1 % (c1 * c2) / c1),
            (e1 / c1 * c1 + e1 % c1, e1),
            (e1 % c1 / c2 * c2 + e1 % c2, e1 % c1, c1 % c2 == 0),
        ]

        extra_bound_patterns = [
            ((ec1, c1, c2), (ec1, c1, c2), lambda ec1, c1, c2: (ec1 * c1) // c2, lambda ec1, c1, c2: ec1 * (c1 // c2)),
        ]

        self.patterns.extend(extra_patterns)
        self.bound_patterns.extend(extra_bound_patterns)


def bound_aware_simplify(exprs: Union[Expr, List[Expr]], var2bound: Dict[Var, Union[BoundInfo, int, Tuple[int, int]]]):
    normalized_var2bound = {}
    for var, bound in var2bound.items():
        if isinstance(bound, int):
            bound = BoundInfo(value=bound)
        elif isinstance(bound, tuple) and len(bound) == 2 and isinstance(bound[0], int) and isinstance(bound[1], int):
            bound = BoundInfo(min_value=bound[0], max_value=bound[1])
        elif isinstance(bound, BoundInfo):
            pass
        else:
            raise ValueError(bound)
        normalized_var2bound[var] = bound
    simplifier = RuleBasedSimplifier(normalized_var2bound)
    return repeat_until_converge(simplifier, exprs)
