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
"""
The module provides a pass that lowers LoadSharedInst to LoadMatrixInst when possible.

We check whether the following conditions to determine whether we can lower a LoadSharedInst to a LoadMatrixInst
    0) the register tensor must have a dtype that can be loaded by a ldmatrix instruction
    1) the layout of the register tensor must be divisible by a ldmatrix layout
    2) the shared tensor address must be aligned with 16 bytes for each row in the ldmatrix unit
    3) each row in the ldmatrix unit must be contiguous in the shared memory
"""

from typing import Optional, Union

from hidet.ir import DataType
from hidet.ir.expr import Expr, Var
from hidet.ir.tools import collect

from tilus import RegisterLayout
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.builders import StmtBuilder
from tilus.ir.func import Analysis, Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import LoadMatrixConfig, LoadSharedInst
from tilus.ir.layout.ops import divide
from tilus.ir.layout.ops.utils import LayoutOperationError
from tilus.ir.stmt import Stmt
from tilus.target import get_current_target, nvgpu_sm75
from tilus.transforms.base import Pass


class LowerToLoadMatrixRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.analysis: Optional[Analysis] = None

    @staticmethod
    def get_load_matrix_config(dtype: DataType, register_layout: RegisterLayout) -> Optional[LoadMatrixConfig]:
        if len(register_layout.shape) != 2:
            return None
        for config in LoadMatrixConfig.all():
            if dtype.nbytes != config.nbytes:
                # condition 0) is not satisfied
                continue
            try:
                divide(register_layout, config.ldmatrix_layout)
            except LayoutOperationError:
                # condition 1) is not satisfied
                continue
            return config
        return None

    def visit_Function(self, func: Function) -> Function:
        self.analysis = func.metadata.analysis
        return super().visit_Function(func)

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> Union[Stmt, Instruction]:
        inst = super().visit_Instruction(inst)

        if not get_current_target().supports(nvgpu_sm75):
            return inst

        register_tensor = inst.register_output
        dtype = register_tensor.dtype

        # determine the load matrix configuration
        config = self.get_load_matrix_config(dtype, register_layout=register_tensor.layout)

        if config is None:
            return inst

        # check the alignment and contiguity of the shared tensor address
        shared_tensor = inst.shared_input
        axes: list[Var] = index_vars(num_vars=len(shared_tensor.shape))
        offset: Expr = shared_tensor.layout(*axes)
        offset_used_vars = collect(offset, [Var], stop_when_found=True)
        var2info = {}
        shape = register_tensor.shape
        for v in offset_used_vars:
            if v in axes:
                continue
            if self.analysis is None:
                continue
            if (
                v in self.analysis.lower_bound
                and v in self.analysis.upper_bound
                and self.analysis.lower_bound[v] == self.analysis.upper_bound[v]
            ):
                var2info[v] = TensorInfo.from_constant(shape=shape, value=self.analysis.lower_bound[v])
            elif v in self.analysis.divisibility:
                var2info[v] = TensorInfo.from_divisibility(shape=shape, divisibility=self.analysis.divisibility[v])
        tensor_info: TensorInfo = analyze_grid(shape=shape, axes=axes, expr=offset, var2info=var2info)

        if tensor_info.infos[-1].divisibility * config.nbytes % 16 != 0:
            # the shared tensor address is not aligned with 16 bytes for each row in the ldmatrix unit
            return inst
        if tensor_info.infos[-1].continuity % config.ldmatrix_layout.shape[-1] != 0:
            # each row in the ldmatrix unit must be contiguous in the shared memory
            return inst

        # we satisfy all the conditions to lower the instruction
        sb = StmtBuilder()
        sb.load_matrix(
            smem_addr=sb.tensor_ptr(shared_tensor, space="shared"),
            axes=axes,
            offset=offset,
            config=config,
            output=inst.register_output,
        )
        return sb.flush_stmts()


class LowerToLoadMatrixPass(Pass):
    def process_function(self, function: Function) -> Function:
        rewriter = LowerToLoadMatrixRewriter()
        return rewriter.visit(function)


def lower_to_load_matrix_pass() -> Pass:
    return LowerToLoadMatrixPass()
