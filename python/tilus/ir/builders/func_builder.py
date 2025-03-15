from typing import List, Union, Optional, Dict, Sequence
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, Var
from hidet.ir.dtypes import int32, boolean
from hidet.ir.primitives.cuda import blockIdx
from tilus.ir.func import Function, BlockMapping, WeightTransform, ParamAttrs
from tilus.ir.stmt import SeqStmt
from tilus.ir.builders.stmt_builder import StatementBuilder


class FunctionContext:
    def __init__(self, builder, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]):
        self.builder: FunctionBuilder = builder
        self.builder._name = name
        self.builder._num_warps = num_warps
        self.builder._params = (
            [Var(name, type) for name, type in params.items()] if isinstance(params, dict) else params
        )
        self.builder._block_axes = []
        self.builder._num_blocks = []
        self.builder._stack = [[]]

    def __enter__(self):
        return self.builder._params

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return

        assert len(self.builder._stack) == 1, len(self.builder._stack)
        axis2size = {a: b for a, b in zip(self.builder._block_axes, self.builder._num_blocks)}
        self.builder._built_function = Function(
            name=self.builder._name,
            params=self.builder._params,
            param2attrs=self.builder._param2attrs,
            num_warps=self.builder._num_warps,
            block_axes=self.builder._block_axes,
            num_blocks=self.builder._num_blocks,
            body=SeqStmt(self.builder._stack.pop()),
            block_mapping=self.create_default_mapping(self.builder._block_axes, axis2size),
            weight_transforms=self.builder._weight_transforms,
            var2divisibility=self.builder._var2divisibility,
            annotations={},
        )

    @staticmethod
    def create_default_mapping(
        block_axes: List[Var],
        axis2size: Dict[Var, Expr],
    ):
        # perform mapping
        logical2hardware: Dict[Var, Var] = {}
        hardware_axes = [blockIdx.x, blockIdx.y, blockIdx.z]

        if len(block_axes) == 0:
            pass
        elif len(block_axes) == 1:
            logical2hardware[block_axes[0]] = blockIdx.x
        elif len(block_axes) == 2:
            logical2hardware[block_axes[0]] = blockIdx.y
            logical2hardware[block_axes[1]] = blockIdx.x
        elif len(block_axes) == 3:
            logical2hardware[block_axes[0]] = blockIdx.z
            logical2hardware[block_axes[1]] = blockIdx.y
            logical2hardware[block_axes[2]] = blockIdx.x
        else:
            for axis in block_axes[:-3]:
                logical2hardware[axis] = blockIdx.z
            logical2hardware[block_axes[-3]] = blockIdx.z
            logical2hardware[block_axes[-2]] = blockIdx.y
            logical2hardware[block_axes[-1]] = blockIdx.x

        # get the logical axis expressions of hardware axes
        hardware_axes_size = {axis: int32.one for axis in hardware_axes}
        for axis in block_axes:
            hardware_axis = logical2hardware[axis]
            hardware_axes_size[hardware_axis] = hardware_axes_size[hardware_axis] * axis2size[axis]

        # get the mapping from logical axis to the expression of hardware axes
        virtual_axes_values = {}
        hardware_axis_divisor = {blockIdx.x: 1, blockIdx.y: 1, blockIdx.z: 1}

        last_virtual_axis: Dict[Var, Var] = {}
        for axis, hardware_axis in logical2hardware.items():
            last_virtual_axis[hardware_axis] = axis

        for axis, hardware_axis in logical2hardware.items():
            virtual_axes_values[axis] = hardware_axis // hardware_axis_divisor[hardware_axis]
            if axis is not last_virtual_axis[hardware_axis]:
                virtual_axes_values[axis] = virtual_axes_values[axis] % axis2size[axis]
            hardware_axis_divisor[hardware_axis] = hardware_axis_divisor[hardware_axis] * axis2size[axis]

        return BlockMapping(
            hardware_axes=hardware_axes,
            hardware_num_blocks=[hardware_axes_size[axis] for axis in hardware_axes],
            predicate=boolean.true,
            virtual_axes_values=virtual_axes_values,
        )


class FunctionBuilder(StatementBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._name: Optional[str] = None
        self._num_warps: Optional[int] = None
        self._params: List[Var] = []
        self._block_axes: List[Var] = []
        self._num_blocks: List[Expr] = []
        self._weight_transforms: Dict[Var, List[WeightTransform]] = {}
        self._param2attrs: Dict[Var, ParamAttrs] = {}
        self._var2divisibility: Dict[Var, int] = {}

        # built function
        self._built_function: Optional[Function] = None

    def __call__(self, *args):
        return self._built_function(*args)

    def __str__(self):
        return str(self._built_function)

    def _reset(self):
        self._stack = [[]]  # for StatementBuilder

        self._name = None
        self._num_warps = None
        self._params = []
        self._block_axes = []
        self._num_blocks = []
        self._weight_transforms = {}
        self._param2attrs = {}
        self._built_function = None

    def virtual_blocks(self, num_blocks: Sequence[Union[Expr, int]]) -> List[Var]:
        self._block_axes = [Var(f"b{i}", int32) for i in range(len(num_blocks))]
        self._num_blocks = [int32(num_block) for num_block in num_blocks]
        return self._block_axes.copy()

    def annotate_divisibility(self, var2divisibility: Dict[Var, int]):
        self._var2divisibility.update(var2divisibility)

    def append_weight_transform(self, param: Var, weight_transform: WeightTransform):
        if param not in self._params:
            raise ValueError(f"Parameter {param} is not defined")
        if param not in self._weight_transforms:
            self._weight_transforms[param] = []
        self._weight_transforms[param].append(weight_transform)

    def set_weight_nbytes(self, param: Var, nbytes: int):
        if param not in self._params:
            raise ValueError(f"Parameter {param} is not defined")
        if param not in self._param2attrs:
            self._param2attrs[param] = ParamAttrs()
        self._param2attrs[param].is_weight = True
        self._param2attrs[param].weight_nbytes = nbytes

    def function(self, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]):
        return FunctionContext(self, name, num_warps, params)

    def flush_function(self) -> Function:
        assert self._built_function is not None
        ret = self._built_function
        self._reset()
        return ret
