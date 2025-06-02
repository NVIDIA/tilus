from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, as_expr

from tilus.extensions.hidet.ir.expr import index_vars
from tilus.extensions.hidet.ir.utils.index_transform import index_multiply
from tilus.ir.node import IRNode
from tilus.utils import prod


@dataclass(frozen=True, eq=False)
class GlobalLayout(IRNode):
    shape: tuple[Expr, ...]
    size: Expr
    axes: tuple[Var, ...]
    offset: Expr

    def __call__(self, *indices: Expr) -> Expr:
        assert len(indices) == len(self.axes)
        from hidet.ir.tools import rewrite

        return rewrite(self.offset, rewrite_map={axis: index for axis, index in zip(self.axes, indices)})

    @staticmethod
    def create(shape: Sequence[Expr | int], size: Expr, f_offset: Callable[[Sequence[Var]], Expr]) -> GlobalLayout:
        expr_shape = tuple(as_expr(s) for s in shape)
        axes: list[Var] = index_vars(num_vars=len(shape))
        return GlobalLayout(shape=expr_shape, size=size, axes=tuple(axes), offset=f_offset(axes))


def _generic_repeat(shape: Sequence[Expr | int], ranks: Sequence[int]) -> GlobalLayout:
    assert len(shape) == len(ranks)
    assert len(ranks) == len(set(ranks)) and all(0 <= d < len(shape) for d in ranks)
    strides: list[Expr] = [prod([s for j, s in enumerate(shape) if ranks[j] > ranks[i]]) for i in range(len(shape))]

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return GlobalLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)


def _global_compose(lhs: GlobalLayout, rhs: GlobalLayout) -> GlobalLayout:
    assert len(lhs.shape) == len(rhs.shape)
    ndims = len(lhs.shape)

    def f_offset(axes: Sequence[Var]) -> Expr:
        lhs_indices = [axes[i] // rhs.shape[i] for i in range(ndims)]
        rhs_indices = [axes[i] // rhs.shape[i] for i in range(ndims)]
        lhs_offset = lhs(*lhs_indices)
        rhs_offset = rhs(*rhs_indices)
        return lhs_offset * rhs.size + rhs_offset

    shape = index_multiply(lhs.shape, rhs.shape)
    size = lhs.size * rhs.size

    return GlobalLayout.create(shape=shape, size=size, f_offset=f_offset)


def global_repeat(*shape: Expr | int) -> GlobalLayout:
    return _generic_repeat(shape=shape, ranks=list(range(len(shape))))


def global_column_repeat(*shape: Expr | int) -> GlobalLayout:
    return _generic_repeat(shape=shape, ranks=list(reversed(range(len(shape)))))


def global_compose(lhs: GlobalLayout, rhs: GlobalLayout, *others: GlobalLayout) -> GlobalLayout:
    if len(others) == 0:
        return _global_compose(lhs, rhs)
    else:
        return global_compose(_global_compose(lhs, rhs), *others)


def global_strides(shape: Sequence[Expr | int], strides: Sequence[Expr | int]) -> GlobalLayout:
    assert len(shape) == len(strides)

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return GlobalLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)
