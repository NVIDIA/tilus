from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import tabulate

from hidet.ir.expr import Expr
from hidet.utils import prod
from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize, index_serialize
from tilus.ir.node import IRNode

Int = int | Expr


class LayoutOperationError(Exception):
    """
    Exception raised when a layout operation fails.
    """

    pass


@dataclass(frozen=True, eq=False)
class RegisterLayout(IRNode):
    shape: tuple[int, ...]
    mode_shape: tuple[int, ...]
    spatial_modes: tuple[int, ...]
    local_modes: tuple[int, ...]

    def __str__(self):
        items = {
            "shape": list(self.shape),
            "mode_shape": list(self.mode_shape),
            "spatial_modes": list(self.spatial_modes),
            "local_modes": list(self.local_modes),
        }
        return "RegisterLayout(" + ", ".join(f"{k}={v}" for k, v in items.items()) + ")"

    def __mul__(self, other):
        if not isinstance(other, RegisterLayout):
            raise TypeError(f"Cannot multiply {type(self)} with {type(other)}")

        from tilus.ir.layout.register_layout_ops import compose

        return compose(self, other)

    def __eq__(self, other):
        if not isinstance(other, RegisterLayout):
            return False
        return (
            self.shape == other.shape
            and self.mode_shape == other.mode_shape
            and self.spatial_modes == other.spatial_modes
            and self.local_modes == other.local_modes
        )

    def with_shape(self, shape: Sequence[int]) -> RegisterLayout:
        validate_layout(shape, self.mode_shape, self.spatial_modes, self.local_modes)
        return dataclasses.replace(self, shape=tuple(shape))

    @cached_property
    def grouped_modes(self):
        i = 0
        grouped_modes = []
        for s in self.shape:
            group_modes = []
            remaining = s
            while remaining > 1:
                remaining //= self.mode_shape[i]
                group_modes.append(i)
                i += 1
            grouped_modes.append(group_modes)
        return grouped_modes

    @cached_property
    def thread_shape(self) -> list[int]:
        return [self.mode_shape[i] if i >= 0 else -i for i in self.spatial_modes]

    @cached_property
    def local_shape(self) -> list[int]:
        return [self.mode_shape[i] for i in self.local_modes]

    def global2local(self, global_indices: Sequence[Int]) -> tuple[list[Int], list[Int]]:
        if len(global_indices) != len(self.shape):
            raise ValueError(
                "Global indices must match the shape of the layout, got {} vs {}".format(
                    len(global_indices), len(self.shape)
                )
            )

        mode_indices: list[Int] = []
        for index, modes in zip(global_indices, self.grouped_modes):
            shape = [self.mode_shape[mode] for mode in modes]
            mode_indices.extend(index_deserialize(index, shape))
        thread_indices: list[Int] = [mode_indices[i] if i >= 0 else 0 for i in self.spatial_modes]
        local_indices: list[Int] = [mode_indices[i] for i in self.local_modes]
        return thread_indices, local_indices

    def local2global(self, thread_indices: Sequence[Int], local_indices: Sequence[Int]) -> list[Int]:
        assert len(thread_indices) == len(self.thread_shape)
        assert len(local_indices) == len(self.local_shape)

        mode_indices: list[Int] = [0 for _ in range(len(self.mode_shape))]
        for i, index in enumerate(thread_indices):
            if self.spatial_modes[i] >= 0:
                mode_indices[self.spatial_modes[i]] = index
        for i, index in enumerate(local_indices):
            mode_indices[self.local_modes[i]] = index

        global_indices: list[Int] = []
        grouped_mode_indices = [[mode_indices[i] for i in group_modes] for group_modes in self.grouped_modes]
        for mode_indices, modes in zip(grouped_mode_indices, self.grouped_modes):
            shape = [self.mode_shape[i] for i in modes]
            global_indices.append(index_serialize(mode_indices, shape))

        return global_indices

    # operations
    def local(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import compose, local

        return compose(self, local(*shape))

    def spatial(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import compose, spatial

        return compose(self, spatial(*shape))

    def column_spatial(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import column_spatial, compose

        return compose(self, column_spatial(*shape))

    def column_local(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import column_local, compose

        return compose(self, column_local(*shape))


def validate_layout(
    shape: Sequence[int],
    mode_shape: Sequence[int],
    spatial_modes: Sequence[int],
    local_modes: Sequence[int],
) -> None:
    """
    Validate the layout parameters.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the layout.

    mode_shape: Sequence[int]
        The shape of the modes.

    spatial_modes: Sequence[int]
        The spatial modes of the layout.

    local_modes: Sequence[int]
        The local modes of the layout.
    """
    assert all(s >= 1 for s in shape), "Shape must only be positive integers"

    # validate modes
    remaining_shape = list(shape)
    for mode in reversed(mode_shape):
        if mode == 1:
            continue

        while remaining_shape and remaining_shape[-1] == 1:
            remaining_shape.pop()

        if len(remaining_shape) == 0 or remaining_shape[-1] % mode != 0:
            raise ValueError(f"Mode {mode} does not divide the remaining shape {remaining_shape}")
        remaining_shape[-1] //= mode
    while remaining_shape and remaining_shape[-1] == 1:
        remaining_shape.pop()

    if remaining_shape:
        raise ValueError("Modes {} and shape {} do not match".format(mode_shape, shape))

    # the thread dims and local dims must be
    # 1. the indices of modes
    # 2. the thread_dims can contain negative values representing replicated threads (containing the same value)
    used_dims = []
    for dim in spatial_modes:
        if dim < 0:
            continue
        if not (0 <= dim < len(mode_shape)):
            raise ValueError(f"Thread dim {dim} is out of range for modes {mode_shape}")
        used_dims.append(dim)
    for dim in local_modes:
        if not (0 <= dim < len(mode_shape)):
            raise ValueError(f"Local dim {dim} is out of range for modes {mode_shape}")
        used_dims.append(dim)
    if len(used_dims) != len(set(used_dims)):
        raise ValueError("Thread dims and local dims must be unique")


def visualize_layout(layout: RegisterLayout) -> str:
    """
    Visualize the layout in a human-readable format.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to be converted.

    Returns
    -------
    ret: str
        The string representation of the layout that is human-readable.
    """
    head = str(layout)

    # normalize the shape into 3-dimension
    shape = [s for s in layout.shape if s > 1]  # prune 1s
    while len(shape) < 3:
        shape.insert(0, 1)
    while len(shape) > 3:
        shape = [prod(shape[0:2])] + shape[2:]

    layout = layout.with_shape(shape)

    tables: list[str] = []
    for batch in range(shape[0]):
        table: list[list[str]] = []
        for i in range(shape[1]):
            row = []
            for j in range(shape[2]):
                thread_indices, local_indices = layout.global2local(global_indices=[batch, i, j])
                replicated_dims = [i for i, dim in enumerate(layout.spatial_modes) if dim < 0]
                replicated_shape = [-layout.spatial_modes[i] for i in replicated_dims]
                threads: list[int] = []
                for replicated_indices in itertools.product(*[range(s) for s in replicated_shape]):
                    for i, idx in zip(replicated_dims, replicated_indices):
                        thread_indices[i] = idx
                    threads.append(int(index_serialize(thread_indices, layout.thread_shape)))
                threads.sort()
                local = int(index_serialize(local_indices, layout.local_shape))
                if len(threads) == 1:
                    row.append(f"{threads[0]}: {local}")
                else:
                    row.append(f"{threads}: {local}")
            table.append(row)
        tables.append(tabulate.tabulate(table, tablefmt="simple_grid"))

    return head + "\n" + "\n".join(tables)


def _canonicalize_singleton_modes(layout: RegisterLayout) -> RegisterLayout:
    singleton_modes = [mode for mode, size in enumerate(layout.mode_shape) if size == 1]
    if not singleton_modes:
        return layout

    mode_map = {}
    i = 0
    for mode, size in enumerate(layout.mode_shape):
        if size == 1:
            mode_map[mode] = -1
        else:
            mode_map[mode] = i
            i += 1

    mode_shape = [size for size in layout.mode_shape if size > 1]
    spatial_modes = [
        mode_map[mode] if mode >= 0 else mode for mode in layout.spatial_modes if mode < 0 or mode_map[mode] != -1
    ]
    local_modes = [mode_map[mode] for mode in layout.local_modes if mode_map[mode] != -1]

    return RegisterLayout(
        shape=layout.shape,
        mode_shape=tuple(mode_shape),
        spatial_modes=tuple(spatial_modes),
        local_modes=tuple(local_modes),
    )


def _canonicalize_contiguous_modes(layout: RegisterLayout) -> RegisterLayout:
    # get the map to mode kind
    mode_kind: dict[int, str] = {}
    mode_index: dict[int, int] = {}
    for i, mode in enumerate(layout.spatial_modes):
        if mode < 0:
            continue
        mode_kind[mode] = "spatial"
        mode_index[mode] = i
    for i, mode in enumerate(layout.local_modes):
        mode_kind[mode] = "local"
        mode_index[mode] = i

    # determine the modes that should be merged
    merge_modes: list[list[int]] = []
    for modes in layout.grouped_modes:
        i = 0
        while i < len(modes):
            j = i
            while (
                j + 1 < len(modes)
                and mode_kind[modes[j]] == mode_kind[modes[j + 1]]
                and mode_index[modes[j]] + 1 == mode_index[modes[j + 1]]
            ):
                j += 1
            merge_modes.append(modes[i : j + 1])
            i = j + 1

    if all(len(modes) == 1 for modes in merge_modes):
        # no merge needed
        return layout

    # get the map from the original mode to the new mode-group, which corresponds to the mode in the canonical layout
    mode_map: dict[int, int] = {}
    for i, modes in enumerate(merge_modes):
        for j, mode in enumerate(modes):
            if j == 0:
                mode_map[mode] = i
            else:
                mode_map[mode] = -1  # this mode is merged with the first mode in the group, mark it as -1

    mode_shape = [prod(layout.mode_shape[i] for i in modes) for modes in merge_modes]
    spatial_modes = [
        mode_map[mode] if mode >= 0 else mode for mode in layout.spatial_modes if mode < 0 or mode_map[mode] != -1
    ]
    local_modes = [mode_map[mode] for mode in layout.local_modes if mode_map[mode] != -1]

    return RegisterLayout(
        shape=layout.shape,
        mode_shape=tuple(mode_shape),
        spatial_modes=tuple(spatial_modes),
        local_modes=tuple(local_modes),
    )


def register_layout(
    shape: Sequence[int],
    mode_shape: Sequence[int],
    spatial_modes: Sequence[int],
    local_modes: Sequence[int],
) -> RegisterLayout:
    """
    Create a register layout with the given shape, mode shape, spatial modes, and local modes.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the layout.

    mode_shape: Sequence[int]
        The shape of the modes.

    spatial_modes: Sequence[int]
        The spatial modes of the layout.

    local_modes: Sequence[int]
        The local modes of the layout.

    Returns
    -------
    ret: RegisterLayout
        The created register layout.
    """
    validate_layout(
        shape=shape,
        mode_shape=mode_shape,
        spatial_modes=spatial_modes,
        local_modes=local_modes,
    )
    layout = RegisterLayout(
        shape=tuple(shape),
        mode_shape=tuple(mode_shape),
        spatial_modes=tuple(spatial_modes),
        local_modes=tuple(local_modes),
    )
    return canonicalize_layout(layout)


def canonicalize_layout(layout: RegisterLayout) -> RegisterLayout:
    """
    Canonicalize the layout by
    1. merging the modes that are contiguous in the three places: shape, spatial_modes, and local_modes
    2. removing the singletons in the modes

    Any layout with the same mapping will be canonicalized to the same layout.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to be canonicalized

    Returns
    -------
    ret: RegisterLayout
        The canonicalized layout
    """
    return _canonicalize_contiguous_modes(_canonicalize_singleton_modes(layout))
