from __future__ import annotations

from typing import Dict, List, Optional, Union

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.ir.type import BaseType
from tilus.extensions.hidet.ir.expr import as_expr
from tilus.ir.builders.stmt_builder import StmtBuilder
from tilus.ir.func import Function, Metadata
from tilus.ir.stmt import SeqStmt


class FunctionBuilder(StmtBuilder):
    class _FunctionContext:
        def __init__(
            self, builder: FunctionBuilder, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]
        ) -> None:
            self.builder: FunctionBuilder = builder
            self.name: str = name
            self.num_warps: int = num_warps
            self.params: List[Var] = (
                [Var(name, type) for name, type in params.items()] if isinstance(params, dict) else params
            )

            self.builder.num_blocks = None
            self.builder._stack = [[]]

        def __enter__(self):
            if len(self.params) == 1:
                return self.params[0]
            else:
                return self.params

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                return

            assert len(self.builder._stack) == 1, len(self.builder._stack)

            if self.builder.num_blocks is None:
                raise ValueError("Please use `builder.num_blocks = ...` to set the number of blocks.")
            if isinstance(self.builder.num_blocks, (int, Expr)):
                num_blocks = [as_expr(self.builder.num_blocks)]
            else:
                num_blocks = [as_expr(item) for item in self.builder.num_blocks]
            if len(num_blocks) > 3:
                raise ValueError("The number of blocks should be at most 3.")
            while len(num_blocks) < 3:
                num_blocks.append(int32.one)

            built_function = Function.create(
                name=self.name,
                params=self.params,
                body=SeqStmt.create(self.builder._stack.pop()),
                metadata=Metadata.create(num_blocks=num_blocks, num_warps=self.num_warps),
            )
            self.builder._on_finish(built_function)

    def __init__(self) -> None:
        super().__init__()
        self.num_blocks: Optional[List[Expr | int] | int | Expr] = None

        # built function
        self._built_function: Optional[Function] = None

    def _reset(self):
        self._stack = [[]]  # for StatementBuilder

        self._name = None
        self._params = []
        self._num_blocks = []

    def _on_finish(self, built_function: Function) -> None:
        self._built_function = built_function
        self._reset()

    def function(self, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]) -> _FunctionContext:
        return self._FunctionContext(self, name, num_warps, params)

    def flush_function(self) -> Function:
        assert self._built_function is not None
        ret = self._built_function
        self._built_function = None
        return ret
