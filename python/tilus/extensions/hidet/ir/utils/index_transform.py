from typing import List, Sequence, Union, Optional, cast as typing_cast
from hidet.ir.expr import Expr
from hidet.ir.dtypes import int32


def index_serialize(
    indices: Sequence[Expr], shape: Sequence[Union[Expr, int]], ranks: Optional[Sequence[int]] = None
) -> Expr:
    """
    Serialize the logical indices in a tensor with given shape to a linear index in linear memory space.
    The ranks indices the rank of each dimension of the tensor.
    ranks = [0, 1, 2, 3] of shape[3, 4, 5, 6] indicates that the last dimension is the fastest changing dimension.
    ranks = [3, 2, 1, 0] of shape[3, 4, 5, 6] indicates that the first dimension is the fastest changing dimension.
    ranks = [0, 2, 1] of shape [3, 4, 5] indicates that the second dimension is the fastest changing dimension.

    In general, the ranks is a permutation of [0, 1, 2, ..., len(shape) - 1]. The dimension with the largest value in
    ranks is the fastest changing dimension. The dimension with the smallest value in ranks is the slowest changing
    dimension.
    """
    if len(shape) == 0:
        return int32.zero
    if ranks is None:
        ranks = list(range(len(shape)))
    scalar_index: Expr = int32.zero
    acc = 1

    for rank in reversed(range(len(shape))):
        assert rank in ranks, f"rank {rank} is not in ranks {ranks}"
        dim = ranks.index(rank)
        idx_value = indices[dim]
        extent = shape[dim]
        scalar_index += idx_value * acc
        acc *= extent
    return scalar_index


def index_deserialize(
    scalar_index: Expr, shape: Sequence[Union[Expr, int]], ranks: Optional[Sequence[int]] = None
) -> List[Expr]:
    """
    reverse of index_serialize
    """
    if len(shape) == 0:
        return []
    if ranks is None:
        ranks = list(range(len(shape)))
    indices: List[Optional[Expr]] = [None for _ in range(len(shape))]
    acc = 1

    for rank in reversed(range(len(shape))):
        assert rank in ranks, f"rank {rank} is not in ranks {ranks}"
        dim = ranks.index(rank)
        extent = shape[dim]
        assert indices[dim] is None, f"index {dim} is already set"

        index = scalar_index

        if rank != len(shape) - 1:
            index = scalar_index // acc

        if rank != 0:
            index = index % extent

        indices[dim] = index

        acc = acc * extent

    assert all(isinstance(idx, Expr) for idx in indices)
    return typing_cast(List[Expr], indices)


def index_add(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]):
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [a + b for a, b in zip(lhs_indices, rhs_indices)]


def index_multiply(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]):
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [a * b for a, b in zip(lhs_indices, rhs_indices)]


def index_mod(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]):
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [a % b for a, b in zip(lhs_indices, rhs_indices)]


def index_divide(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]):
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [a // b for a, b in zip(lhs_indices, rhs_indices)]


def index_sum(indices: Sequence[Union[Expr, int]], init: Union[Expr, int] = 0) -> Union[Expr, int]:
    if len(indices) == 0:
        return init
    else:
        s = indices[0]
        for a in indices[1:]:
            s = s + a
        return s
