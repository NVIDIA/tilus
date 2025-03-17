from typing import List, Dict, Optional, Any
from hidet.ir.expr import Var, Expr
from tilus.ir.stmt import Stmt


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


class Function:
    def __init__(
        self,
        name: str,
        params: List[Var],
        param2attrs: Dict[Var, ParamAttrs],
        num_warps: int,
        num_blocks: List[Expr],
        body: Stmt,
        annotations: Optional[Dict[str, str]],
    ):
        self.name: str = name
        self.params: List[Var] = params
        self.param2attrs: Dict[Var, ParamAttrs] = param2attrs
        self.num_warps: int = num_warps
        self.num_blocks: List[Expr] = num_blocks
        self.body: Stmt = body
        self.annotations: Dict[str, Any] = annotations.copy() if annotations else {}
        self.var2divisibility: Dict[Var, int] = {}

        assert all(isinstance(v, Expr) for v in num_blocks)

    def __str__(self):
        from tilus.ir.tools import IRPrinter

        printer = IRPrinter()
        return str(printer(self))

    def __call__(self, *args):
        module = self.build()
        return module(*args)
