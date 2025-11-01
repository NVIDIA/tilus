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

from typing import Optional, Sequence

from hidet.ir.expr import Expr

from tilus.ir.layout.ops.utils import LayoutOperationError
from tilus.ir.layout.register_layout import (
    RegisterLayout,
    canonicalize_layout,
    register_layout,
)
from tilus.utils import gcd, prod


def spatial(*shape: int, ranks: Optional[Sequence[int]] = None) -> RegisterLayout:
    """
    Create a spatial layout.

    A spatial layout is a layout that maps all dimensions to the spatial dimensions. The ranks of the dimensions are
    specified by the `ranks` parameter.

    Parameters
    ----------
    *shape
        The shape of the layout. Each entry in shape must be a positive constant integer.

    ranks: Sequence[int], optional
        The ranks of the dimensions. The ranks must be unique and in the range [0, len(shape)). If not specified,
        the ranks will be set to the default values [0, 1, 2, ...], indicating the row-major order of the dimensions.

    Returns
    -------
    ret: RegisterLayout
        The spatial layout.
    """
    if ranks is not None:
        if len(shape) != len(ranks):
            raise LayoutOperationError(
                "Shape and ranks must have the same length, got {} vs {}".format(len(shape), len(ranks))
            )
        if len(ranks) != len(set(ranks)):
            raise LayoutOperationError("Ranks must be unique, got {}".format(ranks))
        if any(r < 0 or r >= len(shape) for r in ranks):
            raise LayoutOperationError("Ranks must be in range [0, {}), got {}".format(len(shape), ranks))
        spatial_modes = [ranks.index(i) for i in range(len(shape))]
    else:
        spatial_modes = list(range(len(shape)))
    return register_layout(shape=shape, mode_shape=shape, spatial_modes=spatial_modes, local_modes=[])


def local(*shape: int, ranks: Optional[Sequence[int]] = None) -> RegisterLayout:
    """
    Create a local layout.

    A local layout is a layout that maps all dimensions to the local dimensions. The ranks of the dimensions are
    specified by the `ranks` parameter.

    Parameters
    ----------
    shape:
        The shape of the layout. Each entry in shape must be a positive constant integer.

    ranks: Sequence[int], optional
        The ranks of the dimensions. The ranks must be unique and in the range [0, len(shape)). If not specified,
        the ranks will be set to the default values [0, 1, 2, ...], indicating the row-major order of the dimensions.

    Returns
    -------
    ret: RegisterLayout
        The local layout.
    """
    if ranks is not None:
        if len(shape) != len(ranks):
            raise LayoutOperationError(
                "Shape and ranks must have the same length, got {} vs {}".format(len(shape), len(ranks))
            )
        if len(ranks) != len(set(ranks)):
            raise LayoutOperationError("Ranks must be unique, got {}".format(ranks))
        if any(r < 0 or r >= len(shape) for r in ranks):
            raise LayoutOperationError("Ranks must be in range [0, {}), got {}".format(len(shape), ranks))
        local_modes = [ranks.index(i) for i in range(len(shape))]
    else:
        local_modes = list(range(len(shape)))
    return register_layout(shape=shape, mode_shape=shape, spatial_modes=[], local_modes=local_modes)


def replicated(*shape: int, num_workers: int) -> RegisterLayout:
    """
    Create a full replicated layout.

    A full replicated layout is a layout that replicates the entire tensor across all workers.
    Any worker can access the entire tensor.

    Parameters
    ----------
    *shape:
        The shape of the layout. Each entry in shape must be a positive constant integer.
    num_workers: int
        The number of workers.

    Returns
    -------
    ret: RegisterLayout
        The full replicated layout.
    """
    replicated_unit = RegisterLayout(
        shape=tuple(), mode_shape=tuple(), spatial_modes=(-num_workers,), local_modes=tuple()
    )
    return replicated_unit * local(*shape)


def column_spatial(*shape: int) -> RegisterLayout:
    """
    Create a spatial layout in column-major order.

    Parameters
    ----------
    *shape:
        The shape of the layout. Each entry must be a constant integer.

    Returns
    -------
    ret: RegisterLayout
        The spatial layout.
    """
    return spatial(*shape, ranks=list(reversed(range(len(shape)))))


def column_local(*shape: int) -> RegisterLayout:
    """
    Create a local layout in column-major order.

    Parameters
    ----------
    *shape:
        The shape of the layout. Each entry nust be a constant integer.

    Returns
    -------
    ret: RegisterLayout
        The local layout.
    """
    return local(*shape, ranks=list(reversed(range(len(shape)))))


def squeeze(layout: RegisterLayout, dims: Sequence[int]) -> RegisterLayout:
    """
    Squeeze the layout over the given dimensions.

    The squeeze function will return a new layout with the dimensions specified in dims removed from the layout. The
    specified dimensions must be in the range [0, len(layout.shape)), and the corresponding dimensions in the
    layout must have size 1.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to squeeze.

    dims: Sequence[int]
        The dimensions to squeeze. The dimensions must be in the range [0, len(layout.shape)).

    Returns
    -------
    ret: RegisterLayout
        The squeezed layout.
    """
    if len(dims) == 0:
        return layout
    if any(d < 0 or d >= len(layout.shape) for d in dims):
        raise LayoutOperationError("Dims must be in range [0, {}), got {}".format(len(layout.shape), dims))
    if len(dims) != len(set(dims)):
        raise LayoutOperationError("Dims must be unique, got {}".format(dims))
    if any(layout.shape[d] != 1 for d in dims):
        raise LayoutOperationError("Dims must have size 1, got {}".format([layout.shape[d] for d in dims]))
    shape = [layout.shape[i] for i in range(len(layout.shape)) if i not in dims]
    return layout.with_shape(shape)


def unsqueeze(layout: RegisterLayout, dims: Sequence[int]) -> RegisterLayout:
    """
    Unsqueeze the layout over the given dimensions.

    The unsqueeze function will return a new layout with the dimensions specified in dims added to the layout. The
    new dimensions will have size 1. The dims must be in the range [0, len(layout.shape) + len(dims) - 1],
    representing the dimensions in the new layout.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to unsqueeze.

    dims: Sequence[int]
        The dimensions to unsqueeze.

    Returns
    -------
    ret: RegisterLayout
        The unsqueezed layout.
    """
    if len(dims) == 0:
        return layout
    if any(d < 0 or d >= len(layout.shape) + len(dims) for d in dims):
        raise LayoutOperationError("Dims must be in range [0, {}), got {}".format(len(layout.shape) + len(dims), dims))
    if len(dims) != len(set(dims)):
        raise LayoutOperationError("Dims must be unique, got {}".format(dims))
    shape = []
    current = 0
    for i in range(len(layout.shape) + len(dims)):
        if i in dims:
            shape.append(1)
        else:
            shape.append(layout.shape[current])
            current += 1
    return layout.with_shape(shape)


def compose(outer: RegisterLayout, inner: RegisterLayout) -> RegisterLayout:
    """
    Compose two layouts together.

    Given two layouts with shapes (d_0, d_1, ..., d_{n-1}) and (d'_0, d'_1, ..., d'_{n-1}), the compose function will
    return a new layout with shape (d_0 * d'_0, d_1 * d'_1, ..., d_{n-1} * d'_{n-1}). The modes of the new layout will be
    the concatenation of the modes of the two layouts. The spatial dimensions of the new layout will be the concatenation
    of the spatial dimensions of the two layouts. The local dimensions of the new layout will be the concatenation of the
    local dimensions of the two layouts. The spatial_modes and local_modes of the new layout will be the concatenation of the
    spatial_modes and local_modes of the two layouts, respectively, where the outer layout's spatial_modes and local_modes
    come first, followed by the inner layout's spatial_modes and local_modes.

    Parameters
    ----------
    outer: RegisterLayout
        The outer layout.

    inner: RegisterLayout
        The inner layout.

    Returns
    -------
    ret: RegisterLayout
        The composed layout.
    """
    # unify the ndims of the two layouts
    ndims = max(len(outer.shape), len(inner.shape))
    outer = unsqueeze(outer, [i for i in range(ndims - len(outer.shape))])
    inner = unsqueeze(inner, [i for i in range(ndims - len(inner.shape))])

    # compose the two layouts
    shape: list[int] = [a * b for a, b in zip(outer.shape, inner.shape)]
    outer_map: dict[int, int] = {}  # map from original mode dimension to the new mode dimension
    inner_map: dict[int, int] = {}
    current_outer = 0
    current_inner = 0
    current_composed = 0
    mode_shape: list[int] = []
    for outer_modes, inner_modes in zip(outer.grouped_modes, inner.grouped_modes):
        for outer_mode in outer_modes:
            outer_map[current_outer] = current_composed
            current_outer += 1
            current_composed += 1
            mode_shape.append(outer.mode_shape[outer_mode])
        for inner_mode in inner_modes:
            inner_map[current_inner] = current_composed
            current_inner += 1
            current_composed += 1
            mode_shape.append(inner.mode_shape[inner_mode])
    spatial_modes: list[int] = [outer_map[i] if i >= 0 else i for i in outer.spatial_modes] + [
        inner_map[i] if i >= 0 else i for i in inner.spatial_modes
    ]
    local_modes: list[int] = [outer_map[i] for i in outer.local_modes] + [inner_map[i] for i in inner.local_modes]

    return register_layout(
        shape=shape,
        mode_shape=mode_shape,
        spatial_modes=spatial_modes,
        local_modes=local_modes,
    )


def permute(layout: RegisterLayout, dims: Sequence[int]) -> RegisterLayout:
    """
    Permute the dimensions of the layout.

    Given a layout with shape (d_0, d_1, ..., d_{n-1}), the permute function will return a new layout with shape
    (d_{dims[0]}, d_{dims[1]}, ..., d_{dims[n-1]}).

    Parameters
    ----------
    layout: RegisterLayout
        The layout to permute.

    dims: Sequence[int]
        The permutation order of the dimensions. The length of dims must be equal to the number of dimensions of the
        layout.

    Returns
    -------
    ret: RegisterLayout
        The permuted layout.
    """
    if len(dims) != len(layout.shape):
        raise LayoutOperationError(
            "Dims must have the same length as the layout shape, got {} vs {}".format(len(dims), len(layout.shape))
        )
    if len(dims) != len(set(dims)):
        raise LayoutOperationError("Dims must be unique, got {}".format(dims))
    if any(d < 0 or d >= len(layout.shape) for d in dims):
        raise LayoutOperationError("Dims must be in range [0, {}), got {}".format(len(layout.shape), dims))

    shape = [layout.shape[d] for d in dims]
    grouped_modes = [layout.grouped_modes[d] for d in dims]
    mode_shape = [layout.mode_shape[mode] for group in grouped_modes for mode in group]
    permuted_modes = [i for group in grouped_modes for i in group]

    mode_map = {old: new for new, old in enumerate(permuted_modes)}

    return register_layout(
        shape=shape,
        mode_shape=mode_shape,
        spatial_modes=[mode_map[d] if d >= 0 else d for d in layout.spatial_modes],
        local_modes=[mode_map[d] for d in layout.local_modes],
    )


def reduce(
    layout: RegisterLayout,
    dims: Sequence[int],
    *,
    keepdims: bool = False,
) -> RegisterLayout:
    """
    Reduce the layout over the given dimensions.

    The reduce function will return a new layout with the dimensions specified in dims reduced to 1. The reduced
    dimensions will be removed if `keepdims` is False, or set to 1 if `keepdims` is True.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to reduce.

    dims: Sequence[int]
        The dimensions to reduce. The length of dims must be less than the number of dimensions of the layout.

    keepdims: bool
        Whether to keep the reduced dimensions in the output layout. If True, the reduced dimensions will be set to 1.
        Otherwise, the reduced dimensions will be removed from the output layout.

    Returns
    -------
    ret: RegisterLayout
        The reduced layout.
    """
    if len(dims) == 0:
        return layout
    if any(d < 0 or d >= len(layout.shape) for d in dims):
        raise LayoutOperationError("Dims must be in range [0, {}), got {}".format(len(layout.shape), dims))
    if len(dims) != len(set(dims)):
        raise LayoutOperationError("Dims must be unique, got {}".format(dims))

    mode_map: dict[int, int] = {}
    modes_to_reduce = [mode for i, group in enumerate(layout.grouped_modes) for mode in group if i in dims]
    reduced_modes = [mode for i, group in enumerate(layout.grouped_modes) for mode in group if i not in dims]

    i = 0
    for mode in range(len(layout.mode_shape)):
        if mode in modes_to_reduce:
            mode_map[mode] = -1  # use -1 to indicate that this mode is reduced
        else:
            mode_map[mode] = i
            i += 1

    # compute spatial/local modes
    spatial_modes = []
    for spatial_dim in layout.spatial_modes:
        if spatial_dim < 0:
            spatial_modes.append(spatial_dim)
        else:
            if mode_map[spatial_dim] == -1:
                spatial_modes.append(-layout.mode_shape[spatial_dim])
            else:
                spatial_modes.append(mode_map[spatial_dim])
    local_modes = [mode_map[d] for d in layout.local_modes if mode_map[d] != -1]

    if keepdims:
        shape = [layout.shape[d] if d not in dims else 1 for d in range(len(layout.shape))]
    else:
        shape = [layout.shape[d] for d in range(len(layout.shape)) if d not in dims]
    mode_shape = [layout.mode_shape[mode] for mode in reduced_modes]

    return register_layout(
        shape=shape,
        mode_shape=mode_shape,
        spatial_modes=spatial_modes,
        local_modes=local_modes,
    )


def reduce_to(
    layout: RegisterLayout,
    shape: Sequence[int],
) -> RegisterLayout:
    reduce_dims_keep_dims = []
    reduce_dims = []

    diff_rank = len(layout.shape) - len(shape)
    for i in range(len(layout.shape)):
        if i < diff_rank:
            reduce_dims.append(i)
        elif layout.shape[i] != shape[i - diff_rank] and shape[i - diff_rank] == 1:
            reduce_dims_keep_dims.append(i)
        elif layout.shape[i] == shape[i - diff_rank]:
            # matched dimension
            continue
        else:
            raise ValueError("can not broadcast output layout from input shape")

    if reduce_dims_keep_dims:
        layout = reduce(layout, reduce_dims_keep_dims, keepdims=True)
    if reduce_dims:
        layout = reduce(layout, reduce_dims, keepdims=False)
    return layout


def concat(lhs: RegisterLayout, rhs: RegisterLayout) -> RegisterLayout:
    """
    Concatenate two layouts.

    The concat function will return a new layout with the dimensions of the two layouts concatenated.

    Parameters
    ----------
    lhs: RegisterLayout
        The left-hand side layout.
    rhs: RegisterLayout
        The right-hand side layout.

    Returns
    -------
    ret: RegisterLayout
        The concatenated layout.
    """
    lhs_num_modes = len(lhs.mode_shape)
    return register_layout(
        shape=lhs.shape + rhs.shape,
        mode_shape=lhs.mode_shape + rhs.mode_shape,
        spatial_modes=lhs.spatial_modes + tuple(i + lhs_num_modes if i >= 0 else i for i in rhs.spatial_modes),
        local_modes=lhs.local_modes + tuple(i + lhs_num_modes for i in rhs.local_modes),
    )


def reshape(layout: RegisterLayout, shape: Sequence[int]) -> RegisterLayout:
    """
    Reshape the layout to the given shape.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to reshape.

    shape: Sequence[int]
        The shape to reshape to. The shape must be compatible with the layout's shape.

    Returns
    -------
    ret: RegisterLayout
        The reshaped layout.
    """
    if prod(layout.shape) != prod(shape):
        raise LayoutOperationError("Cannot reshape layout with shape {} to shape {}".format(layout.shape, shape))

    original_shape = tuple(shape)

    # canonicalize the layout to make sure the contiguous modes in both shape and spatial/local are merged
    layout = canonicalize_layout(layout)

    # check the new shape only redistributes/splitting the existing modes
    mode_shape = list(layout.mode_shape)
    shape = list(shape)
    grouped_mode_shape: list[list[int]] = []  # for the new layout

    while mode_shape:
        p = mode_shape.pop(0)
        grouped_mode_shape.append([])

        while shape:
            q = shape[0]
            if q % p == 0:
                grouped_mode_shape[-1].append(p)
                shape[0] = q // p
                break
            elif p % q == 0:
                if q > 1:
                    grouped_mode_shape[-1].append(q)
                shape.pop(0)
            else:
                raise LayoutOperationError("Cannot reshape layout {} to shape {}".format(layout, shape))

    new_mode_shape = [size for group in grouped_mode_shape for size in group]
    mode_map: dict[int, list[int]] = {}

    i = 0
    for mode, mode_shape in enumerate(grouped_mode_shape):
        mode_map[mode] = [i + j for j in range(len(mode_shape))]
        i += len(mode_shape)

    spatial_modes = []
    for mode in layout.spatial_modes:
        if mode < 0:
            spatial_modes.append(mode)
        else:
            spatial_modes.extend(mode_map[mode])

    local_modes = []
    for mode in layout.local_modes:
        local_modes.extend(mode_map[mode])

    return register_layout(
        shape=original_shape,
        mode_shape=new_mode_shape,
        spatial_modes=spatial_modes,
        local_modes=local_modes,
    )


def flatten(layout: RegisterLayout, start_dim: int = 0, end_dim: int = -1) -> RegisterLayout:
    """
    Flatten the layout over the dimensions between start_dim and end_dim, inclusive.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to flatten.

    start_dim: int
        The start dimension to flatten over. The dimension is inclusive.

    end_dim: int
        The end dimension to flatten over. The dimension is inclusive. If end_dim is -1, it will be set to the last
        dimension of the layout.

    Returns
    -------
    ret: RegisterLayout
        The flattened layout.
    """
    if end_dim < 0:
        end_dim = len(layout.shape) + end_dim
    shape = layout.shape[:start_dim] + (prod(layout.shape[start_dim:end_dim]),) + layout.shape[end_dim:]
    return reshape(layout, shape)


def _layout_with_mode_shape(layout: RegisterLayout, mode_shape: Sequence[int]) -> RegisterLayout:
    # transform the layout to use the given mode shape, it must be a valid and fine-grained mode shape than
    # the original one
    assert prod(layout.shape) == prod(mode_shape)
    mode_map: dict[int, list[int]] = {}  # map from original mode dimension to the new mode dimensions
    i = 0
    for mode, original_mode_size in enumerate(layout.mode_shape):
        mode_map[mode] = []
        while i < len(mode_shape) and original_mode_size % mode_shape[i] == 0:
            mode_map[mode].append(i)
            original_mode_size //= mode_shape[i]
            i += 1
        assert original_mode_size == 1

    # expand the original spatial and local modes
    spatial_modes = []
    for mode in layout.spatial_modes:
        if mode < 0:
            spatial_modes.append(mode)
        else:
            spatial_modes.extend(mode_map[mode])

    local_modes = []
    for mode in layout.local_modes:
        local_modes.extend(mode_map[mode])

    return RegisterLayout(  # do not use register_layout here, because the returned layout should not be canonicalized
        shape=layout.shape,
        mode_shape=tuple(mode_shape),
        spatial_modes=tuple(spatial_modes),
        local_modes=tuple(local_modes),
    )


def divide(lhs: RegisterLayout, rhs: RegisterLayout) -> RegisterLayout:
    """
    Divide two layouts.

    Given two layouts lhs and rhs, the divide function will return a new layout result, such that:

    lhs = compose(result, rhs)

    If no such layout exists, the function will raise a LayoutOperationFailed exception.

    Parameters
    ----------
    lhs: RegisterLayout
        The layout to be divided.
    rhs: RegisterLayout
        The layout to divide by.

    Returns
    -------
    ret: RegisterLayout
        The result layout.
    """
    # 0. refine the mode_shape of the lhs layout to make it compatible with the rhs layout
    # 1. check whether we can divide the two layouts
    #  1.1. the rhs layout's grouped modes must be a suffix of the lhs layout's grouped modes for each group
    #  1.2. all such suffix dimension must also be the suffix of the spatial/local modes in lhs layout
    # 2. construct the result of division lhs / rhs
    #  2.1. calculate the mode_shape of the result layout
    #  2.2. construct a mapping from the original mode in lhs layout to the new mode in the result layout
    #  2.2. calculate the new spatial/local modes of the result layout based on the mapping

    if len(lhs.shape) < len(rhs.shape):
        raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
    if len(lhs.shape) > len(rhs.shape):
        rhs = unsqueeze(rhs, [i for i in range(len(lhs.shape) - len(rhs.shape))])

    lhs = canonicalize_layout(lhs)
    rhs = canonicalize_layout(rhs)

    # 0. refine the mode_shape of the lhs layout to make it compatible with the rhs layout
    mode_shape = []
    for lhs_group, rhs_group in zip(lhs.grouped_modes, rhs.grouped_modes):
        if len(lhs_group) < len(rhs_group):
            raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
        refined_group_shape = [lhs.mode_shape[mode] for mode in lhs_group[: len(lhs_group) - len(rhs_group)]]
        for i, (p, q) in enumerate(zip(lhs_group[-len(rhs_group) :], rhs_group)):
            p, q = lhs.mode_shape[p], rhs.mode_shape[q]
            if p == q:
                refined_group_shape.append(p)
                continue
            else:
                if i == 0 and p % q == 0:
                    refined_group_shape.append(p // q)
                    refined_group_shape.append(q)
                else:
                    raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
        mode_shape.extend(refined_group_shape)
    lhs = _layout_with_mode_shape(lhs, mode_shape)

    # 1.1. check the grouped modes
    lhs_grouped_modes = lhs.grouped_modes
    rhs_grouped_modes = rhs.grouped_modes

    assert len(lhs_grouped_modes) == len(rhs_grouped_modes)

    for i in range(len(lhs_grouped_modes)):
        lhs_group = lhs_grouped_modes[i]
        rhs_group = rhs_grouped_modes[i]
        if len(lhs_group) < len(rhs_group):
            raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
        for a, b in zip(lhs_group[-len(rhs_group) :], rhs_group):
            if lhs.mode_shape[a] != rhs.mode_shape[b]:
                raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))

    # 1.2. check the spatial/local modes
    mode_map = {}  # from rhs mode to its corresponding mode in lhs layout
    i = 0
    j = 0
    for lhs_mode_group, rhs_mode_group in zip(lhs.grouped_modes, rhs.grouped_modes):
        j += len(lhs_mode_group) - len(rhs_mode_group)
        for k in range(len(rhs_mode_group)):
            mode_map[i] = j
            i += 1
            j += 1
    if len(lhs.spatial_modes) < len(rhs.spatial_modes) or len(lhs.local_modes) < len(rhs.local_modes):
        raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
    for lhs_mode, rhs_mode in zip(lhs.spatial_modes[-len(rhs.spatial_modes) :], rhs.spatial_modes):
        if lhs_mode == rhs_mode < 0:
            continue
        if lhs_mode < 0 or rhs_mode < 0:
            raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
        if mode_map[rhs_mode] != lhs_mode:
            raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
    for lhs_mode, rhs_mode in zip(lhs.local_modes[-len(rhs.local_modes) :], rhs.local_modes):
        if mode_map[rhs_mode] != lhs_mode:
            raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))
    for lhs_mode, rhs_mode in zip(lhs.local_modes[-len(rhs.local_modes) :], rhs.local_modes):
        if mode_map[rhs_mode] != lhs_mode:
            raise LayoutOperationError("Cannot divide layout {} by layout {}".format(lhs, rhs))

    # 2.1. calculate the mode_shape of the result layout
    shape = [a // b for a, b in zip(lhs.shape, rhs.shape)]
    mode_shape = []
    for lhs_group, rhs_group in zip(lhs.grouped_modes, rhs.grouped_modes):
        pruned_lhs_group = lhs_group[: -len(rhs_group)] if len(rhs_group) > 0 else lhs_group
        pruned_lhs_shape = [lhs.mode_shape[mode] for mode in pruned_lhs_group]
        mode_shape.extend(pruned_lhs_shape)

    # 2.2 construct a mapping from the original mode in lhs layout to the new mode in the result layout
    mode_map = {}
    i = 0
    for lhs_group, rhs_group in zip(lhs.grouped_modes, rhs.grouped_modes):
        pruned_lhs_group = lhs_group[: -len(rhs_group)] if len(rhs_group) > 0 else lhs_group
        for lhs_mode in pruned_lhs_group:
            mode_map[lhs_mode] = i
            i += 1

    # 2.3 calculate the new spatial/local modes of the result layout based on the mapping
    pruned_lhs_spatial_modes = (
        lhs.spatial_modes[: -len(rhs.spatial_modes)] if len(rhs.spatial_modes) > 0 else lhs.spatial_modes
    )
    spatial_modes = [mode_map[mode] if mode >= 0 else mode for mode in pruned_lhs_spatial_modes]
    pruned_lhs_local_modes = lhs.local_modes[: -len(rhs.local_modes)] if len(rhs.local_modes) > 0 else lhs.local_modes
    local_modes = [mode_map[mode] for mode in pruned_lhs_local_modes]

    # return the result layout
    return register_layout(
        shape=shape,
        mode_shape=mode_shape,
        spatial_modes=spatial_modes,
        local_modes=local_modes,
    )


def left_divide(layout: RegisterLayout, lhs_divisor: RegisterLayout) -> RegisterLayout:
    """
    Divide the layout from the left by the lhs_divisor.

    Given two layouts: layout and lhs_divisor, the left_divide function will return a new layout result, such that:

    layout = compose(lhs_divisor, result)

    If no such layout exists, the function will raise a LayoutOperationFailed exception.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to be divided.
    lhs_divisor: RegisterLayout
        The layout to divide by.

    Returns
    -------
    ret: RegisterLayout
        The result layout.
    """
    # 0. refine the mode_shape of the layout to make it compatible with the lhs_divisor layout
    # 1. check whether we can left-divide the two layouts
    #  1.1. the lhs_divisor layout's grouped modes must be a prefix of the layout's grouped modes for each group
    #  1.2. all such prefix dimensions must also be the prefix of the spatial/local modes in layout
    # 2. construct the result of left division: layout = compose(lhs_divisor, result)
    #  2.1. calculate the mode_shape of the result layout (remove the prefix modes)
    #  2.2. construct a mapping from the original mode in layout to the new mode in the result layout
    #  2.3. calculate the new spatial/local modes of the result layout based on the mapping

    if len(layout.shape) < len(lhs_divisor.shape):
        raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))
    if len(layout.shape) > len(lhs_divisor.shape):
        lhs_divisor = unsqueeze(lhs_divisor, [i for i in range(len(layout.shape) - len(lhs_divisor.shape))])

    layout = canonicalize_layout(layout)
    lhs_divisor = canonicalize_layout(lhs_divisor)

    # 0. refine the mode_shape of the layout to make it compatible with the lhs_divisor layout (prefix alignment)
    refined_mode_shape: list[int] = []
    for lay_group, div_group in zip(layout.grouped_modes, lhs_divisor.grouped_modes):
        lay_sizes = [layout.mode_shape[m] for m in lay_group]
        div_sizes = [lhs_divisor.mode_shape[m] for m in div_group]

        group_refined: list[int] = []
        i = 0  # pointer in lay_sizes
        for j, q in enumerate(div_sizes):
            if i >= len(lay_sizes):
                raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))
            p = lay_sizes[i]
            if p == q:
                group_refined.append(p)
                i += 1
            else:
                if j == 0 and p % q == 0:
                    # split the first element so that prefix matches
                    group_refined.append(q)
                    lay_sizes[i] = p // q
                    if lay_sizes[i] == 1:
                        # keep 1 as a mode piece to be consistent with divide's refinement behavior
                        pass
                else:
                    raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))
        # append the remaining parts of the layout group
        group_refined.extend(lay_sizes[i:])
        refined_mode_shape.extend(group_refined)
    layout = _layout_with_mode_shape(layout, refined_mode_shape)

    # 1.1. check the grouped modes (prefix)
    lay_grouped = layout.grouped_modes
    div_grouped = lhs_divisor.grouped_modes
    assert len(lay_grouped) == len(div_grouped)
    for i in range(len(lay_grouped)):
        lay_group = lay_grouped[i]
        div_group = div_grouped[i]
        if len(lay_group) < len(div_group):
            raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))
        for a, b in zip(lay_group[: len(div_group)], div_group):
            if layout.mode_shape[a] != lhs_divisor.mode_shape[b]:
                raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))

    # 1.2. check the spatial/local modes (prefix)
    # build mapping from lhs_divisor mode to corresponding mode index in layout
    mode_map: dict[int, int] = {}
    i = 0  # mode index in lhs_divisor
    j = 0  # mode index in layout
    for lay_group, div_group in zip(layout.grouped_modes, lhs_divisor.grouped_modes):
        for _ in range(len(div_group)):
            mode_map[i] = j
            i += 1
            j += 1
        j += len(lay_group) - len(div_group)

    if len(layout.spatial_modes) < len(lhs_divisor.spatial_modes) or len(layout.local_modes) < len(
        lhs_divisor.local_modes
    ):
        raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))

    for lay_mode, div_mode in zip(layout.spatial_modes[: len(lhs_divisor.spatial_modes)], lhs_divisor.spatial_modes):
        if lay_mode == div_mode < 0:
            continue
        if lay_mode < 0 or div_mode < 0:
            raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))
        if mode_map[div_mode] != lay_mode:
            raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))

    for lay_mode, div_mode in zip(layout.local_modes[: len(lhs_divisor.local_modes)], lhs_divisor.local_modes):
        if mode_map[div_mode] != lay_mode:
            raise LayoutOperationError("Cannot left-divide layout {} by layout {}".format(layout, lhs_divisor))

    # 2.1. calculate the mode_shape of the result layout (remove prefix)
    shape = [a // b for a, b in zip(layout.shape, lhs_divisor.shape)]
    mode_shape: list[int] = []
    for lay_group, div_group in zip(layout.grouped_modes, lhs_divisor.grouped_modes):
        pruned_lay_group = lay_group[len(div_group) :] if len(div_group) > 0 else lay_group
        mode_shape.extend([layout.mode_shape[m] for m in pruned_lay_group])

    # 2.2. mapping from original layout modes to result modes (after removing prefix)
    result_mode_map: dict[int, int] = {}
    new_index = 0
    for lay_group, div_group in zip(layout.grouped_modes, lhs_divisor.grouped_modes):
        pruned_lay_group = lay_group[len(div_group) :] if len(div_group) > 0 else lay_group
        for m in pruned_lay_group:
            result_mode_map[m] = new_index
            new_index += 1

    # 2.3. spatial/local modes for result (remove prefix then remap)
    pruned_spatial = (
        layout.spatial_modes[len(lhs_divisor.spatial_modes) :]
        if len(lhs_divisor.spatial_modes) > 0
        else layout.spatial_modes
    )
    spatial_modes = [result_mode_map[m] if m >= 0 else m for m in pruned_spatial]
    pruned_local = (
        layout.local_modes[len(lhs_divisor.local_modes) :] if len(lhs_divisor.local_modes) > 0 else layout.local_modes
    )
    local_modes = [result_mode_map[m] for m in pruned_local]
    return register_layout(
        shape=shape,
        mode_shape=mode_shape,
        spatial_modes=spatial_modes,
        local_modes=local_modes,
    )


def auto_local_spatial(num_threads: int, shape: Sequence[int]) -> RegisterLayout:
    """Create a local(...).spatial(...) layout

    This function automatically determines a composition of the local and spatial layouts, based on the number of threads
    and the shape of the composed layout.

    Parameters
    ----------
    num_threads: int
        The number of threads to be used for the spatial layout. This should be a positive integer.
    shape: Sequence[int]
        The shape of the composed layout. Each entry in shape must be a positive constant integer.

    Returns
    -------
    ret: RegisterLayout
        The layout that is a composition of local and spatial layouts.
    """
    size = prod(shape)
    assert size % num_threads == 0 or num_threads % size == 0, (
        "Cannot auto local spatial layout with shape {} and num_threads {}".format(shape, num_threads)
    )

    remain_shape = list(shape)
    remain_threads = num_threads
    spatial_shape = [1 for i in range(len(shape))]

    for i in reversed(range(len(shape))):
        spatial_shape[i] = gcd(remain_threads, remain_shape[i])
        remain_threads //= spatial_shape[i]
        remain_shape[i] //= spatial_shape[i]

    local_shape = remain_shape
    ret = local(*local_shape).spatial(*spatial_shape)

    if remain_threads != 1:
        ret = register_layout(
            shape=ret.shape,
            mode_shape=ret.mode_shape,
            spatial_modes=(-remain_threads,) + ret.spatial_modes,
            local_modes=ret.local_modes,
        )

    return ret


def locate_at(layout: RegisterLayout, global_indices: Sequence[Expr | int], spatial_index: Expr | int) -> Expr:
    """
    Check if the global indices are located at the given spatial index.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to be checked.

    global_indices: Sequence[Int]
        The global indices to be checked.

    spatial_index: Int
        The spatial index to be checked.

    Returns
    -------
    ret: Expr
        Expression with value True if the global indices are located at the given spatial index, False otherwise.
    """
    from hidet.ir.dtypes import boolean
    from hidet.ir.expr import logical_and

    from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize

    if len(global_indices) != len(layout.shape):
        raise ValueError(
            "Global indices must match the shape of the layout, got {} vs {}".format(
                len(global_indices), len(layout.shape)
            )
        )

    mode_indices: list[Expr | int] = []
    for index, modes in zip(global_indices, layout.grouped_modes):
        shape = [layout.mode_shape[mode] for mode in modes]
        mode_indices.extend(index_deserialize(index, shape))

    condition = boolean.true

    spatial_indices: list[Expr] = index_deserialize(spatial_index, layout.spatial_shape)
    for i, mode in enumerate(layout.spatial_modes):
        if mode < 0:
            continue
        condition = logical_and(condition, mode_indices[mode] == spatial_indices[i])
    return condition
