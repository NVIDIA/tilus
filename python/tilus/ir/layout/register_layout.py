from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import tabulate
from hidet import boolean
from hidet.ir.expr import Expr, logical_and
from hidet.utils import prod

from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize, index_serialize
from tilus.ir.layout.mfunction import MultiFunction, multi_function
from tilus.ir.node import IRNode

Int = int | Expr


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

    def __truediv__(self, other):
        if not isinstance(other, RegisterLayout):
            raise TypeError(f"Cannot divide {type(self)} with {type(other)}")

        from tilus.ir.layout.register_layout_ops import divide

        return divide(self, other)

    def __eq__(self, other):
        if not isinstance(other, RegisterLayout):
            return False
        return (
            self.shape == other.shape
            and self.mode_shape == other.mode_shape
            and self.spatial_modes == other.spatial_modes
            and self.local_modes == other.local_modes
        )

    def __hash__(self):
        return id(self)

    def with_shape(self, shape: Sequence[int]) -> RegisterLayout:
        validate_layout(shape, self.mode_shape, self.spatial_modes, self.local_modes)
        return dataclasses.replace(self, shape=tuple(shape))

    @cached_property
    def grouped_modes(self):
        from .utils import get_mode_groups

        return get_mode_groups(self.shape, self.mode_shape)

    @cached_property
    def spatial_shape(self) -> list[int]:
        return [self.mode_shape[i] if i >= 0 else -i for i in self.spatial_modes]

    @cached_property
    def local_shape(self) -> list[int]:
        return [self.mode_shape[i] for i in self.local_modes]

    @cached_property
    def local_size(self) -> int:
        return prod(self.local_shape)

    @cached_property
    def spatial_size(self) -> int:
        return prod(self.spatial_shape)

    @cached_property
    def size(self) -> int:
        return prod(self.shape)

    def spatial_mfunction(self) -> MultiFunction:
        """
        Get the multi-function that maps the global indices to the spatial indices (serialized).
        """
        return multi_function(
            shape=self.shape,
            mode_shape=self.mode_shape,
            modes=self.spatial_modes,
        )

    def get_spatial(self, global_indices: Sequence[Int]) -> list[Expr]:
        mode_indices: list[Int] = []
        for index, modes in zip(global_indices, self.grouped_modes):
            shape = [self.mode_shape[mode] for mode in modes]
            mode_indices.extend(index_deserialize(index, shape))

        replicate_dims = []
        replicate_sizes = []
        spatial_indices: list[Int] = []
        for i, mode in enumerate(self.spatial_modes):
            if mode < 0:
                replicate_dims.append(i)
                replicate_sizes.append(-mode)
                spatial_indices.append(0)
            else:
                spatial_indices.append(mode_indices[mode])

        results: list[Expr] = []
        for items in itertools.product(*[range(s) for s in replicate_sizes]):
            for dim, value in zip(replicate_dims, items):
                spatial_indices[dim] = value
            results.append(index_serialize(spatial_indices, self.spatial_shape))
        return results

    def get_local(self, global_indices: Sequence[Int]) -> Expr:
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
        local_indices: list[Int] = [mode_indices[i] for i in self.local_modes]
        return index_serialize(local_indices, self.local_shape)

    def get_global(self, *, spatial_index: Int, local_index: Int) -> list[Expr]:
        spatial_indices = index_deserialize(spatial_index, self.spatial_shape)
        local_indices = index_deserialize(local_index, self.local_shape)

        mode_indices: list[Int] = [0 for _ in range(len(self.mode_shape))]
        for i, index in enumerate(spatial_indices):
            if self.spatial_modes[i] >= 0:
                mode_indices[self.spatial_modes[i]] = index
        for i, index in enumerate(local_indices):
            mode_indices[self.local_modes[i]] = index

        global_indices: list[Expr] = []
        grouped_mode_indices = [[mode_indices[i] for i in group_modes] for group_modes in self.grouped_modes]
        for mode_indices, modes in zip(grouped_mode_indices, self.grouped_modes):
            shape = [self.mode_shape[i] for i in modes]
            global_indices.append(index_serialize(mode_indices, shape))

        return global_indices

    # operations
    def repeat(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import compose, repeat

        return compose(self, repeat(*shape))

    def spatial(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import compose, spatial

        return compose(self, spatial(*shape))

    def column_spatial(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import column_spatial, compose

        return compose(self, column_spatial(*shape))

    def column_local(self, *shape: int) -> RegisterLayout:
        from tilus.ir.layout.register_layout_ops import column_repeat, compose

        return compose(self, column_repeat(*shape))

    def reduce_to(self, shape: Sequence[int]) -> RegisterLayout:
        """
        Reduce the layout to the given shape by removing the modes that are not in the shape.

        Parameters
        ----------
        shape: Sequence[int]
            The shape to reduce to.

        Returns
        -------
        ret: RegisterLayout
            The reduced layout.
        """
        from tilus.ir.layout.register_layout_ops import reduce_to

        return reduce_to(self, shape)


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
                local_index = layout.get_local(global_indices=[batch, i, j])
                thread_indices = layout.get_spatial(global_indices=[batch, i, j])
                thread_indices.sort()
                if len(thread_indices) == 1:
                    row.append(f"{thread_indices[0]}: {local_index}")
                else:
                    row.append(f"{thread_indices}: {local_index}")
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


def locate_at(layout: RegisterLayout, global_indices: Sequence[Int], spatial_index: Int) -> Expr:
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
    if len(global_indices) != len(layout.shape):
        raise ValueError(
            "Global indices must match the shape of the layout, got {} vs {}".format(
                len(global_indices), len(layout.shape)
            )
        )

    mode_indices: list[Int] = []
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
