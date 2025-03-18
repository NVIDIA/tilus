from typing import List, Union, Optional, Dict
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, Var
from tilus.ir.func import Function
from tilus.ir.stmt import SeqStmt
from tilus.ir.builders.stmt_builder import StatementBuilder


class FunctionBuilder(StatementBuilder):
    class _FunctionContext:
        def __init__(self, builder, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]):
            self.builder: FunctionBuilder = builder
            self.builder._name = name
            self.builder._num_warps = num_warps
            self.builder._params = (
                [Var(name, type) for name, type in params.items()] if isinstance(params, dict) else params
            )
            self.builder._block_axes = []
            self.builder._num_blocks = []
            self.builder._stack = [[]]

        def __enter__(self):
            return self.builder._params

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                return

            assert len(self.builder._stack) == 1, len(self.builder._stack)
            built_function = Function(
                name=self.builder._name,
                params=self.builder._params,
                # param2attrs=self.builder._param2attrs,
                num_warps=self.builder._num_warps,
                num_blocks=self.builder._num_blocks,
                body=SeqStmt(self.builder._stack.pop()),
                annotations={},
            )
            self.builder._on_finish(built_function)

    def __init__(self) -> None:
        super().__init__()
        self._name: Optional[str] = None
        self._num_warps: Optional[int] = None
        self._params: List[Var] = []
        self._block_axes: List[Var] = []
        self._num_blocks: List[Expr] = []
        # self._param2attrs: Dict[Var, ParamAttrs] = {}
        self._var2divisibility: Dict[Var, int] = {}

        # built function
        self._built_function: Optional[Function] = None

    def _reset(self):
        self._stack = [[]]  # for StatementBuilder

        self._name = None
        self._num_warps = None
        self._params = []
        self._block_axes = []
        self._num_blocks = []
        self._weight_transforms = {}
        self._param2attrs = {}

    def _on_finish(self, built_function: Function):
        self._built_function = built_function
        self._reset()

    def function(self, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]):
        return self._FunctionContext(self, name, num_warps, params)

    def flush_function(self) -> Function:
        assert self._built_function is not None
        ret = self._built_function
        self._built_function = None
        return ret
