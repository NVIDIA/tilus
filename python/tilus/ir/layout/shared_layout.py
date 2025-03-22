from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.utils import prod
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.extensions.hidet.ir.utils.index_transform import vector_mul
from tilus.ir.node import IRNode


@dataclass(frozen=True, eq=False)
class SharedLayout(IRNode):
    shape: tuple[int, ...]
    size: int
    axes: tuple[Var, ...]
    offset: Expr

    def __call__(self, *indices: Expr) -> Expr:
        assert len(indices) == len(self.axes)
        from hidet.ir.tools import rewrite

        return rewrite(self.offset, rewrite_map={axis: index for axis, index in zip(self.axes, indices)})

    @staticmethod
    def create(shape: Sequence[int], size: int, f_offset: Callable[[Sequence[Var]], Expr]) -> SharedLayout:
        axes: List[Var] = index_vars(num_vars=len(shape))
        return SharedLayout(shape=tuple(shape), size=size, axes=tuple(axes), offset=f_offset(axes))

    def simplify(self) -> SharedLayout:
        from tilus.extensions.hidet.transforms.rule_based_simplifier import BoundInfo, RuleBasedSimplifier

        var2bound: Dict[Var, BoundInfo] = {
            axis: BoundInfo(min_value=0, max_value=extent - 1) for axis, extent in zip(self.axes, self.shape)
        }
        simplifier = RuleBasedSimplifier(var2bound=var2bound)
        return SharedLayout(shape=self.shape, size=self.size, axes=self.axes, offset=simplifier(self.offset))

    def swizzle(self, dim: int, regards_dim: int, log_step: int) -> SharedLayout:
        ndims = len(self.shape)
        assert 0 <= dim < ndims and 0 <= regards_dim < ndims and dim != regards_dim

        def get_xor_index(indices: Sequence[Expr]) -> Expr:
            indices = list(indices)  # copy
            step = 2**log_step
            regards_index = indices[regards_dim] // step
            regards_extent = self.shape[regards_dim] // step
            if regards_extent > self.shape[dim]:
                regards_index = regards_index % self.shape[dim]
            return regards_index

        def f_offset(axes: Sequence[Var]) -> Expr:
            swizzled_indices: List[Expr] = [axis for axis in axes]
            swizzled_indices[dim] = swizzled_indices[dim] ^ get_xor_index(axes)
            return self(*swizzled_indices)

        return SharedLayout.create(shape=self.shape, size=self.size, f_offset=f_offset)

    def prepend_dim(self, extent: int) -> SharedLayout:
        def f_offset(axes: Sequence[Var]) -> Expr:
            tile_offset = axes[0] * self.size
            return tile_offset + self(*axes[1:])

        return SharedLayout.create(shape=(extent,) + self.shape, size=extent * self.size, f_offset=f_offset)

    def unsqueeze(self, dims: Sequence[int]) -> SharedLayout:
        shape = []
        cur_dim = 0
        for i in range(len(self.shape) + len(dims)):
            if i in dims:
                shape.append(1)
            else:
                shape.append(self.shape[cur_dim])
                cur_dim += 1

        def f_offset(axes: Sequence[Var]) -> Expr:
            base_axes = [axis for i, axis in enumerate(axes) if i not in dims]
            return self(*base_axes)

        return SharedLayout.create(shape=shape, size=self.size, f_offset=f_offset)


def _generic_repeat(shape: List[int], ranks: List[int]) -> SharedLayout:
    assert len(shape) == len(ranks)
    assert len(ranks) == len(set(ranks)) and all(0 <= d < len(shape) for d in ranks)
    strides: List[int] = [prod([s for j, s in enumerate(shape) if ranks[j] > ranks[i]]) for i in range(len(shape))]

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return SharedLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)


def _shared_compose(lhs: SharedLayout, rhs: SharedLayout) -> SharedLayout:
    assert len(lhs.shape) == len(rhs.shape)
    ndims = len(lhs.shape)

    def f_offset(axes: Sequence[Var]) -> Expr:
        lhs_axes = [axes[i] // rhs.shape[i] for i in range(ndims)]
        rhs_axes = [axes[i] % rhs.shape[i] for i in range(ndims)]
        lhs_offset = lhs(*lhs_axes)
        rhs_offset = rhs(*rhs_axes)
        return lhs_offset * rhs.size + rhs_offset

    shape = vector_mul(lhs.shape, rhs.shape)
    size = lhs.size * rhs.size

    return SharedLayout.create(shape=shape, size=size, f_offset=f_offset)


def shared_repeat(*shape: int) -> SharedLayout:
    return _generic_repeat(shape=list(shape), ranks=list(range(len(shape))))


def shared_column_repeat(*shape: int) -> SharedLayout:
    return _generic_repeat(shape=list(shape), ranks=list(reversed(range(len(shape)))))


def shared_compose(lhs: SharedLayout, rhs: SharedLayout, *others: SharedLayout) -> SharedLayout:
    if len(others) == 0:
        return _shared_compose(lhs, rhs)
    else:
        return shared_compose(_shared_compose(lhs, rhs), *others)
