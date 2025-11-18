# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Sequence, Union
from typing import cast as typing_cast

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, as_expr, logical_and


def index_serialize(
    indices: Sequence[Expr | int], shape: Sequence[Union[Expr, int]], ranks: Optional[Sequence[int]] = None
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
    if len(indices) != len(shape):
        raise ValueError(f"Expect indices length {len(indices)} to match shape length {len(shape)}")
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
    scalar_index: Expr | int, shape: Sequence[Union[Expr, int]], ranks: Optional[Sequence[int]] = None
) -> List[Expr]:
    """
    Reverse of index_serialize
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

        index = as_expr(scalar_index)

        if rank != len(shape) - 1:
            index = as_expr(scalar_index) // acc

        if rank != 0:
            index = index % extent

        indices[dim] = index

        acc = acc * extent

    assert all(isinstance(idx, Expr) for idx in indices)
    return typing_cast(List[Expr], indices)


def index_add(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]) -> List[Expr]:
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [as_expr(a) + as_expr(b) for a, b in zip(lhs_indices, rhs_indices)]


def index_multiply(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]) -> List[Expr]:
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [as_expr(a) * as_expr(b) for a, b in zip(lhs_indices, rhs_indices)]


def index_mod(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]) -> List[Expr]:
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [as_expr(a) % as_expr(b) for a, b in zip(lhs_indices, rhs_indices)]


def index_divide(lhs_indices: Sequence[Union[Expr, int]], rhs_indices: Sequence[Union[Expr, int]]) -> List[Expr]:
    assert len(lhs_indices) == len(rhs_indices), "Expect both indices have the same length"
    return [as_expr(a) // as_expr(b) for a, b in zip(lhs_indices, rhs_indices)]


def index_sum(indices: Sequence[Union[Expr, int]], init: Union[Expr, int] = 0) -> Expr:
    if len(indices) == 0:
        return as_expr(init)
    else:
        s: Expr = as_expr(indices[0])
        for a in indices[1:]:
            s = s + as_expr(a)
        return s


def index_within_bound(
    indices: Sequence[Expr | int],
    lower_bound: Sequence[Expr | int] | Expr | int,
    upper_bound: Sequence[Expr | int] | Expr | int,
) -> Expr:
    # check if the indices are within the bound
    if isinstance(lower_bound, (int, Expr)):
        lower_bound = [lower_bound for _ in indices]
    if isinstance(upper_bound, (int, Expr)):
        upper_bound = [upper_bound for _ in indices]
    assert len(indices) == len(lower_bound) == len(upper_bound), "Expect all indices have the same length"
    conditions = [
        logical_and(lower <= idx, idx < upper) for lower, idx, upper in zip(lower_bound, indices, upper_bound)
    ]
    return logical_and(*conditions)


def vector_mul(lhs: Sequence[int], rhs: Sequence[int] | int) -> List[int]:
    if isinstance(rhs, int):
        rhs = [rhs for _ in lhs]
    assert len(lhs) == len(rhs), "Expect both indices have the same length"
    return [a * b for a, b in zip(lhs, rhs)]


def vector_add(lhs: Sequence[int], rhs: Sequence[int] | int) -> List[int]:
    if isinstance(rhs, int):
        rhs = [rhs for _ in lhs]
    assert len(lhs) == len(rhs), "Expect both indices have the same length"
    return [a + b for a, b in zip(lhs, rhs)]


def vector_sub(lhs: Sequence[int], rhs: Sequence[int] | int) -> List[int]:
    if isinstance(rhs, int):
        rhs = [rhs for _ in lhs]
    assert len(lhs) == len(rhs), "Expect both indices have the same length"
    return [a - b for a, b in zip(lhs, rhs)]


def vector_div(lhs: Sequence[int], rhs: Sequence[int] | int) -> List[int]:
    if isinstance(rhs, int):
        rhs = [rhs for _ in lhs]
    assert len(lhs) == len(rhs), "Expect both indices have the same length"
    return [a // b for a, b in zip(lhs, rhs)]


def vector_within_bound(
    indices: Sequence[int],
    lower_bound: Sequence[int] | int,
    upper_bound: Sequence[int] | int,
) -> bool:
    # check if the indices are within the bound
    if isinstance(lower_bound, int):
        lower_bound = [lower_bound for _ in indices]
    if isinstance(upper_bound, int):
        upper_bound = [upper_bound for _ in indices]
    assert len(indices) == len(lower_bound) == len(upper_bound), "Expect all indices have the same length"
    conditions = [lower <= idx < upper for lower, idx, upper in zip(lower_bound, indices, upper_bound)]
    return all(conditions)
