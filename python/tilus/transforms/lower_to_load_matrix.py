"""
This module provides a pass that lowers LoadSharedInst to LoadMatrixInst when possible.

We check whether the following conditions to determine whether we can lower a LoadSharedInst to a LoadMatrixInst
    0) the register tensor must have a dtype that can be loaded by a ldmatrix instruction
    1) the layout of the register tensor must be divisible by a ldmatrix layout
    2) the shared tensor address must be aligned with 16 bytes for each row in the ldmatrix unit
    3) each row in the ldmatrix unit must be contiguous in the shared memory
"""

from typing import Optional, Union

from hidet.ir import DataType
from hidet.ir.expr import Expr, Var
from tilus import RegisterLayout
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.builders import StmtBuilder
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import LoadMatrixConfig, LoadSharedInst
from tilus.ir.layout import divide
from tilus.ir.stmt import Stmt
from tilus.target import get_current_target, nvgpu_sm75
from tilus.transforms.base import Pass


class LowerToLoadMatrixRewriter(IRRewriter):
    @staticmethod
    def get_load_matrix_config(dtype: DataType, register_layout: RegisterLayout) -> Optional[LoadMatrixConfig]:
        if len(register_layout.shape) != 2:
            return None
        for config in LoadMatrixConfig.all():
            if dtype.nbytes != config.nbytes:
                # condition 0) is not satisfied
                continue
            outer_layout: Optional[RegisterLayout] = divide(register_layout, config.ldmatrix_layout)
            if outer_layout is None:
                # condition 1) is not satisfied
                continue
            return config
        return None

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
        tensor_info: TensorInfo = analyze_grid(shape=register_tensor.shape, axes=axes, expr=offset, var2info={})

        if tensor_info.infos[-1].divisibility * config.nbytes % 16 != 0:
            # the shared tensor address is not aligned with 16 bytes for each row in the ldmatrix unit
            return inst
        if tensor_info.infos[-1].continuity % config.ldmatrix_layout.shape[-1] != 0:
            # each row in the ldmatrix unit must be contiguous in the shared memory
            return inst

        # we satisfy all the conditions to lower the instruction
        sb = StmtBuilder()
        sb.load_matrix(
            ptr=sb.tensor_ptr(shared_tensor), axes=axes, offset=offset, config=config, output=inst.register_output
        )
        return sb.flush_stmts()


class LowerToLoadMatrixPass(Pass):
    def process_function(self, function: Function) -> Function:
        rewriter = LowerToLoadMatrixRewriter()
        return rewriter.visit(function)


def lower_to_load_matrix_pass() -> Pass:
    return LowerToLoadMatrixPass()
