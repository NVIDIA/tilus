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
    """The layout for global tensor.

    Attributes
    ----------
    shape: tuple[Expr, ...]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a constant
        integer.
    size: Expr
        The storage size of the global tensor, in number of elements. If the layout is a `compact` layout, size
        should be equal to the product of the shape dimensions. Otherwise, it can be either larger (in case of padding)
        or smaller (in case of sharing data for different elements) than the product of the shape dimensions.
    axes: tuple[Var, ...]
        The axes of the global tensor. Each axis is a variable that represents the index of the corresponding dimension.
        It should have the same length as the shape.
    offset: Expr
        An expression that computes the offset of the global tensor based on the axes. Only the axes and grid-invariant
        variables can be used in the expression.
    """

    shape: tuple[Expr, ...]
    size: Expr
    axes: tuple[Var, ...]
    offset: Expr

    def __call__(self, *indices: Expr) -> Expr:
        """Compute the offset on given indices.

        This method computes the offset of an element in the global tensor with the given indices.

        Parameters
        ----------
        indices: Sequence[Expr]
            The indices of the global tensor. The length of the indices should match the number of axes in the layout.

        Returns
        -------
        ret: Expr
            The computed offset of the global tensor element at the given indices.
        """
        assert len(indices) == len(self.axes)
        from hidet.ir.tools import rewrite

        return rewrite(self.offset, rewrite_map={axis: index for axis, index in zip(self.axes, indices)})

    @staticmethod
    def create(shape: Sequence[Expr | int], size: Expr, f_offset: Callable[[Sequence[Var]], Expr]) -> GlobalLayout:
        """Create a global layout with custom mapping.

        This method creates a global layout with the given shape, size, and the mapping function for the offset.

        Parameters
        ----------
        shape: Sequence[Expr | int]
            The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
            constant integer.
        size: Expr
            The storage size of the global tensor, in number of elements. If the layout is a `compact` layout, size
            should be equal to the product of the shape dimensions. Otherwise, it can be either larger (in case of
            padding) or smaller (in case of sharing data for different elements) than the product of the shape dimensions.
        f_offset: Callable[[Sequence[Var]], Expr]
            A function that takes a sequence of axes (variables) and returns an expression that computes the offset
            of the global tensor based on those axes. The axes are variables that represent the index of the corresponding
            dimension. The function should not use any variables that are not grid-invariant.

        Returns
        -------
        ret: GlobalLayout
            A new instance of GlobalLayout with the specified shape, size, axes, and offset function.
        """
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


def global_row_major(*shape: Expr | int) -> GlobalLayout:
    """Create a global layout with row-major order.

    Parameters
    ----------
    shape: Sequence[Expr | int]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
        constant integer.

    Returns
    -------
    ret: GlobalLayout
        A global layout with the specified shape in row-major order.
    """
    return _generic_repeat(shape=shape, ranks=list(range(len(shape))))


def global_column_major(*shape: Expr | int) -> GlobalLayout:
    """Create a global layout with column-major order.

    Parameters
    ----------
    shape: Sequence[Expr | int]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
        constant integer.

    Returns
    -------
    ret: GlobalLayout
        A global layout with the specified shape in column-major order.
    """
    return _generic_repeat(shape=shape, ranks=list(reversed(range(len(shape)))))


def global_compose(lhs: GlobalLayout, rhs: GlobalLayout, *others: GlobalLayout) -> GlobalLayout:
    """Compose multiple global layouts.

    This function composes two or more global layouts into a single global layout.
    Please refer to our research paper `Tilus <https://arxiv.org/pdf/2504.12984>`_, Section 4.2 for more details on layout composition.

    Parameters
    ----------
    lhs: GlobalLayout
        The left-hand side global layout.
    rhs: GlobalLayout
        The right-hand side global layout.
    others: Sequence[GlobalLayout]
        The additional global layouts to be composed with the first two. It's optional and can be empty.

    Returns
    -------
    ret: GlobalLayout
        The composed global layout that combines the effects of all input layouts.
    """
    if len(others) == 0:
        return _global_compose(lhs, rhs)
    else:
        return global_compose(_global_compose(lhs, rhs), *others)


def global_strides(shape: Sequence[Expr | int], strides: Sequence[Expr | int]) -> GlobalLayout:
    """Create a global layout with specified strides.

    This function creates a global layout with the given shape and strides. Given the axes and strides, we map the
    axes to ``sum(axes[i] * strides[i])`` to compute the offset of the global tensor.

    Parameters
    ----------
    shape: Sequence[Expr | int]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
        constant integer.
    strides: Sequence[Expr | int]
        The strides of the global tensor. Each stride corresponds to the step size in each dimension when traversing
        the global tensor. It should have the same length as the shape. Each stride can be an expression of grid-invariant
        expression, or a constant integer.

    Returns
    -------
    ret: GlobalLayout
        A global layout with the specified shape and strides. The offset is computed as the sum of the product of each
        axis and its corresponding stride.
    """
    assert len(shape) == len(strides)

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return GlobalLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)
