import dataclasses

from tilus.ir.node import IRNode


@dataclasses.dataclass(frozen=True, eq=False)
class TMemLayout(IRNode):
    shape: tuple[int, int]
    mode_shape: tuple[int, ...]
    lane_modes: tuple[int, ...]
    column_modes: tuple[int, ...]
