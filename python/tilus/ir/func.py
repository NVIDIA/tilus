from typing import List, Dict, Optional, Any
from hidet.ir.expr import Var, Expr
from tilus.ir.stmt import Stmt
from tilus.ir.weight_transform import WeightTransform


class ParamAttrs:
    def __init__(
        self,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
        divisibility: Optional[int] = None,
        is_weight: bool = False,
        weight_nbytes: Optional[int] = None,
    ):
        self.lower: Optional[int] = lower
        self.upper: Optional[int] = upper
        self.divisibility: Optional[int] = divisibility
        self.is_weight: bool = is_weight
        self.weight_nbytes: Optional[int] = weight_nbytes

    def __str__(self) -> str:
        items: List[str] = []
        if self.lower is not None:
            items.append("lower={}".format(self.lower))
        if self.upper is not None:
            items.append("upper={}".format(self.upper))
        if self.divisibility is not None:
            items.append("divisibility={}".format(self.divisibility))
        if self.is_weight:
            items.append("is_weight=True")
            if self.weight_nbytes is not None:
                items.append("weight_nbytes={}".format(self.weight_nbytes))
        return ", ".join(items)

    def is_nontrivial(self):
        return self.upper is not None or self.lower is not None or self.divisibility is not None or self.is_weight


class BlockMapping:
    def __init__(
        self,
        hardware_axes: List[Var],
        hardware_num_blocks: List[Expr],
        predicate: Expr,
        virtual_axes_values: Dict[Var, Expr],
    ):
        # the hardware block axes
        self.hardware_axes: List[Var] = hardware_axes
        # the extent of each hardware axis
        self.hardware_num_blocks: List[Expr] = hardware_num_blocks
        # whether the given hardware block axes should participate the computation
        self.predicate: Expr = predicate
        # when predicate evaluates to True, how each virtual axis (block axes and inter block reduce axes) are
        # calculated given the hardware axes
        self.virtual_axes_values: Dict[Var, Expr] = virtual_axes_values


class Function:
    def __init__(
        self,
        name: str,
        params: List[Var],
        param2attrs: Dict[Var, ParamAttrs],
        num_warps: int,
        block_axes: List[Var],
        num_blocks: List[Expr],
        body: Stmt,
        block_mapping: BlockMapping,
        weight_transforms: Optional[Dict[Var, List[WeightTransform]]],
        var2divisibility: Optional[Dict[Var, int]],
        annotations: Optional[Dict[str, str]],
    ):
        self.name: str = name
        self.params: List[Var] = params
        self.param2attrs: Dict[Var, ParamAttrs] = param2attrs
        self.num_warps: int = num_warps
        self.block_axes: List[Var] = block_axes
        self.num_blocks: List[Expr] = num_blocks
        self.body: Stmt = body
        self.block_mapping: BlockMapping = block_mapping
        self.weight_transforms: Dict[Var, List[WeightTransform]] = weight_transforms if weight_transforms else {}
        self.var2divisibility: Dict[Var, int] = (
            var2divisibility.copy() if var2divisibility else {}
        )  # todo: make compiler analyze this
        self.annotations: Dict[str, Any] = annotations.copy() if annotations else {}

        assert block_mapping is not None
        assert all(isinstance(v, Expr) for v in num_blocks)

    def __str__(self):
        from tilus.ir.tools import IRPrinter

        printer = IRPrinter()
        return str(printer(self))

    def __call__(self, *args):
        module = self.build()
        return module(*args)
