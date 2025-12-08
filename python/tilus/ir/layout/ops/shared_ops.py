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
from __future__ import annotations

from typing import List, Sequence

import tabulate
from hidet.utils import gcd, prod

from tilus.ir.layout.ops.utils import LayoutOperationError, get_mode_groups
from tilus.ir.layout.shared_layout import SharedLayout, Swizzle, shared_layout


def strides_from_ranks(shape: Sequence[int], ranks: Sequence[int]) -> list[int]:
    """
    Compute the strides from the ranks of each dimension.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the tensor.
    ranks: Sequence[int]
        The ranks of each dimension. The length of ranks must be equal to the length of shape
        and all elements in ranks must be unique and in the range [0, len(shape)).

    Returns
    -------
    ret: list[int]
        The strides of each dimension.
    """
    assert len(shape) == len(ranks)
    assert len(ranks) == len(set(ranks)) and all(0 <= d < len(shape) for d in ranks)
    strides: list[int] = [prod([s for j, s in enumerate(shape) if ranks[j] > ranks[i]]) for i in range(len(shape))]
    return strides


def shared_row_major(*shape: int) -> SharedLayout:
    """Create a shared layout with row-major order.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the shared tensor. Each dimension is a constant integer.

    Returns
    -------
    ret: SharedLayout
        A shared layout with the specified shape in row-major order.
    """
    mode_shape = shape
    mode_strides = strides_from_ranks(shape=mode_shape, ranks=list(range(len(mode_shape))))
    return shared_layout(shape=shape, mode_shape=mode_shape, mode_strides=mode_strides, optional_swizzle=None)


def shared_column_major(*shape: int) -> SharedLayout:
    """Create a shared layout with column-major order.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the shared tensor. Each dimension is a constant integer.

    Returns
    -------
    ret: SharedLayout
        A shared layout with the specified shape in column-major order.
    """
    mode_shape = shape
    mode_strides = strides_from_ranks(shape=mode_shape, ranks=list(reversed(range(len(mode_shape)))))
    return shared_layout(shape=shape, mode_shape=mode_shape, mode_strides=mode_strides, optional_swizzle=None)


def shared_compose(lhs: SharedLayout, rhs: SharedLayout) -> SharedLayout:
    """Compose multiple shared layouts together.

    Parameters
    ----------
    lhs: SharedLayout
        The first shared layout to compose.
    rhs: SharedLayout
        The second shared layout to compose.

    Returns
    -------
    ret: SharedLayout
        The composed shared layout.
    """
    assert len(lhs.shape) == len(rhs.shape)
    ndims = len(lhs.shape)

    # shape
    shape = tuple(lhs.shape[i] * rhs.shape[i] for i in range(ndims))

    # mode shape
    lhs_mode_groups = get_mode_groups(lhs.shape, lhs.mode_shape)
    rhs_mode_groups = get_mode_groups(rhs.shape, rhs.mode_shape)
    mode_shape: list[int] = []
    for lhs_group, rhs_group in zip(lhs_mode_groups, rhs_mode_groups):
        mode_shape.extend([lhs.mode_shape[i] for i in lhs_group])
        mode_shape.extend([rhs.mode_shape[i] for i in rhs_group])

    # mode strides
    mode_strides: list[int] = []
    rhs_size = rhs.count_size()
    for lhs_group, rhs_group in zip(lhs_mode_groups, rhs_mode_groups):
        mode_strides.extend([stride * rhs_size for stride in (lhs.mode_strides[i] for i in lhs_group)])
        mode_strides.extend([rhs.mode_strides[i] for i in rhs_group])

    return shared_layout(shape=shape, mode_shape=mode_shape, mode_strides=mode_strides, optional_swizzle=None)


def shared_permute(layout: SharedLayout, dims: Sequence[int]) -> SharedLayout:
    """Permute the dimensions of the shared layout.

    Parameters
    ----------
    layout: SharedLayout
        The layout to permute.

    dims: Sequence[int]
        The permutation order of the dimensions. The length of dims must be equal to the number of dimensions of the
        layout.

    Returns
    -------
    ret: SharedLayout
        The permuted layout.
    """
    assert len(dims) == len(layout.shape) and set(dims) == set(range(len(layout.shape)))

    # shape
    shape = tuple(layout.shape[d] for d in dims)

    # mode shape and mode strides
    layout_mode_groups = get_mode_groups(layout.shape, layout.mode_shape)
    mode_shape: list[int] = []
    mode_strides: list[int] = []
    for d in dims:
        mode_shape.extend([layout.mode_shape[i] for i in layout_mode_groups[d]])
        mode_strides.extend([layout.mode_strides[i] for i in layout_mode_groups[d]])

    return shared_layout(
        shape=shape, mode_shape=mode_shape, mode_strides=mode_strides, optional_swizzle=layout.optional_swizzle
    )


def shared_slice(layout: SharedLayout, retain_dims: Sequence[int]) -> SharedLayout:
    """Slice the shared layout by removing specified dimensions.

    Parameters
    ----------
    layout: SharedLayout
        The layout to slice.
    dims: Sequence[int]
        The dimensions to slice. Each dimension should be in the range [0, len(layout.shape)). The dimensions will
        be kept in the output layout.

    Returns
    -------
    ret: SharedLayout
        The sliced layout.
    """
    assert all(0 <= d < len(layout.shape) for d in retain_dims) and len(retain_dims) == len(set(retain_dims))
    shape: List[int] = []
    mode_shape: List[int] = []
    mode_strides: List[int] = []
    layout_mode_groups = get_mode_groups(layout.shape, layout.mode_shape)
    for i in retain_dims:
        shape.append(layout.shape[i])
        mode_shape.extend([layout.mode_shape[j] for j in layout_mode_groups[i]])
        mode_strides.extend([layout.mode_strides[j] for j in layout_mode_groups[i]])

    return shared_layout(
        shape=shape,
        mode_shape=mode_shape,
        mode_strides=mode_strides,
        optional_swizzle=layout.optional_swizzle,
    )


def shared_unsqueeze(layout: SharedLayout, dims: Sequence[int]) -> SharedLayout:
    """Unsqueeze the shared layout by adding new dimensions of size 1.

    Parameters
    ----------
    layout: SharedLayout
        The layout to unsqueeze.
    dims: Sequence[int]
        The dimensions to unsqueeze. Each dimension should be in the range [0, len(layout.shape)].

    Returns
    -------
    ret: SharedLayout
        The unsqueezed layout.
    """
    assert all(0 <= d <= len(layout.shape) for d in dims) and len(dims) == len(set(dims))
    shape: List[int] = list(layout.shape)
    for d in sorted(dims):
        shape.insert(d, 1)
    return shared_layout(
        shape=shape,
        mode_shape=layout.mode_shape,
        mode_strides=layout.mode_strides,
        optional_swizzle=layout.optional_swizzle,
    )


def shared_row_major_swizzle(shape: Sequence[int], dtype_nbytes: int) -> SharedLayout:
    """
    Generate a shared layout that could be used to generate ldmatrix instruction when using LoadSharedInst.

    Both m and n must be a multiple of 8.

    We will divide each row into bank groups, and bank group has 16 bytes (16 x uint8, 8 x fp16, or 4 x fp32, etc.).
    They correspond to 4 banks in shared memory. For example, if we have m = n = 8 and dtype=fp16, we can represent
    bank groups as

    0   # bank group 0, banks from 0 to 3
    1   # bank group 1, banks from 4 to 7
    2   # ...
    3
    4
    5
    6
    7   # bank groups 7, banks from 28 to 31

    Given m, and n, we need to find a proper way to organize the m x (n / 8) bank groups in shared memory, so that
    1) each row has different bank groups
    2) each column has different bank groups

    When we have m = 8 and n = 64, we have 8 x 8 bank groups. If we store the elements in row-major order, we will
    have the bank groups as

    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7
    0  1  2  3  4  5  6  7

    If we use ldmatrix to load the above 8 x 64 shared memory, we will need 8 ldmatrix.v1 instructions. Each instruction
    loads one column (8 x 8 elements, or 8 x 1 bank groups). Since each instruction will access the same bank group,
    severe bank conflicts will occur. Thus, we need to change the layout of shared memory to avoid bank conflicts.

    Let layout(i, j) be the shared memory address of logical elements (each element has 16 bytes) when we use
    a specific `layout`. For example, the row-major layout row-major(i, j) = i * n + j * 8 (we assume the dtype has 2
    bytes). If we use the swizzled layout swizzled(i, j) = row-major(i, j ^ i) = i * n + (j ^ i) * 8, we can have the
    following bank groups in shared memory.

    0  1  2  3  4  5  6  7
    1  0  3  2  5  4  7  6
    2  3  0  1  6  7  4  5
    3  2  1  0  7  6  5  4
    4  5  6  7  0  1  2  3
    5  4  7  6  1  0  3  2
    6  7  4  5  2  3  0  1
    7  6  5  4  3  2  1  0

    (reader may need some time to figure out the above layout...)

    This layout has two benefits:
    1) Each row has different bank groups. In above example, we have 32 banks per row.
    2) Each column has different bank groups. In above example, we have 32 banks per column.

    The benefit 1 makes sure that when we load data from global memory to shared memory, we can store efficiently.
    The benefit 2 makes sure that when we load data from shared memory to register memory, we can load efficiently.

    We can always generate the swizzled layout for arbitrary m and n as long as they are multiple of 8. See the
    implementation for more details.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the shared memory. The shape must have at least two dimensions.

    dtype_nbytes: int
        The element data type size in bytes.

    Returns
    -------
    shared_layout: SharedLayout
        The shared layout that could be used to generate ldmatrix instruction when using LoadSharedInst.
    """
    if len(shape) < 2:
        raise ValueError("The shape of swizzled shared layout must have at least two dimensions.")
    head, m, n = tuple(shape[:-2]), shape[-2], shape[-1]

    if m % 8 != 0 or n * dtype_nbytes % 16 != 0:
        raise ValueError("m must be a multiple of 8, and n * dtype_nbytes must be a multiple of 16.")

    n_vector_size: int = gcd(n, 128 // dtype_nbytes)
    n_num_vectors: int = n // n_vector_size

    mode_shape = head + (m, n_num_vectors, n_vector_size)

    # use the order of head, columns_vectors, rows, columns_vec_size to compute the strides
    ranks = list(range(len(head))) + [len(head) + 1, len(head), len(head) + 2]
    mode_strides = strides_from_ranks(shape=mode_shape, ranks=ranks)

    log2 = {
        1: 0,
        2: 1,
        4: 2,
        8: 3,
        16: 4,
    }

    if n_vector_size * dtype_nbytes == 128:
        """
        (each number represents a 16-byte group of elements)
        0  1  2  3  4  5  6  7
        1  0  3  2  5  4  7  6
        2  3  0  1  6  7  4  5
        3  2  1  0  7  6  5  4
        4  5  6  7  0  1  2  3
        5  4  7  6  1  0  3  2
        6  7  4  5  2  3  0  1
        7  6  5  4  3  2  1  0
        """
        swizzle = Swizzle(base=log2[16 // dtype_nbytes], bits=3, shift=3)
    elif n_vector_size * dtype_nbytes == 64:
        """
        0  1  2  3
        4  5  6  7
        1  0  3  2
        5  4  7  6
        2  3  0  1
        6  7  4  5
        3  2  1  0
        7  6  5  4
        """
        swizzle = Swizzle(base=log2[16 // dtype_nbytes], bits=2, shift=3)
    elif n_vector_size * dtype_nbytes == 32:
        """
        0  1
        2  3
        4  5
        6  7
        1  0
        3  2
        5  4
        7  6
        """
        swizzle = Swizzle(base=log2[16 // dtype_nbytes], bits=1, shift=3)
    elif n_vector_size * dtype_nbytes == 16:
        """
        0
        1
        2
        3
        4
        5
        6
        7
        """
        swizzle = None
    else:
        assert False

    return shared_layout(
        shape=shape,
        mode_shape=mode_shape,
        mode_strides=mode_strides,
        optional_swizzle=swizzle,
    )


def visualize_layout(layout: SharedLayout, tablefmt: str = "simple_grid") -> str:
    """
    Visualize the layout in a human-readable format.

    Parameters
    ----------
    layout: SharedLayout
        The layout to be converted.

    tablefmt: str
        The table format to use. It should be a valid format specifier in tabulate.tabulate function.
        Candidates:

        - simple_grid
        - plain
        - grid
        - rounded_grid
        - mixed_grid
        - double_grid
        - fancy_grid
        - outline
        - simple_outline
        - mixed_outline
        - presto

    Returns
    -------
    ret: str
        The string representation of the layout that is human-readable.
    """
    head = str(layout)
    assert isinstance(layout, SharedLayout)

    if len(layout.shape) != 2:
        raise LayoutOperationError(f"Shared layout with shape {layout.shape} is not supported for visualization.")
    grid = layout.as_numpy_grid()
    table = []
    for i in range(layout.shape[0]):
        row = []
        for j in range(layout.shape[1]):
            row.append(f"{grid[i, j]}")
        table.append(row)
    return head + "\n" + tabulate.tabulate(table, tablefmt=tablefmt)
