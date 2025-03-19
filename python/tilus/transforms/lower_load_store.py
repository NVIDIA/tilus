from typing import Any, Sequence

from hidet.ir.expr import Expr, Var
from tilus.extensions.hidet.ir.utils.index_transform import index_within_bound
from tilus.ir.builders import StmtBuilder
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import LoadGlobalInst, StoreGlobalInst
from tilus.ir.layout import GlobalLayout
from tilus.ir.stmt import Stmt
from tilus.transforms.base import Pass


class LowerLoadStoreRewriter(IRRewriter):
    @staticmethod
    def get_funcs(offsets: tuple[Expr, ...], dims: tuple[int, ...], global_layout: GlobalLayout) -> tuple[Any, Any]:
        def f_global_indices(indices: Sequence[Var]) -> list[Expr]:
            global_indices: list[Expr] = list(offsets)
            for i, dim in enumerate(sorted(dims)):
                global_indices[dim] = global_indices[dim] + indices[i]
            return global_indices

        def f_offset(indices: Sequence[Var]) -> Expr:
            return global_layout(*f_global_indices(indices))

        def f_mask(indices: Sequence[Var]) -> Expr:
            global_indices = f_global_indices(indices)
            return index_within_bound(global_indices, 0, global_layout.shape)

        return f_offset, f_mask

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> Stmt:
        inst = super().default_visit_Instruction(inst)

        sb = StmtBuilder()
        global_tensor = inst.inputs[0].as_global_tensor()
        register_tensor = inst.register_output
        ptr = sb.tensor_ptr(global_tensor)

        f_offset, f_mask = self.get_funcs(offsets=inst.offsets, dims=inst.dims, global_layout=global_tensor.layout)

        self.memo[inst.register_output] = sb.load_global_generic(
            dtype=global_tensor.dtype, layout=register_tensor.layout, ptr=ptr, f_offset=f_offset, f_mask=f_mask
        )
        return sb.flush_stmts()

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> Stmt:
        inst = super().default_visit_Instruction(inst)

        sb = StmtBuilder()
        global_tensor = inst.inputs[0].as_global_tensor()
        register_tensor = inst.inputs[1].as_register_tensor()
        ptr = sb.tensor_ptr(global_tensor)

        f_offset, f_mask = self.get_funcs(offsets=inst.offsets, dims=inst.dims, global_layout=global_tensor.layout)

        sb.store_global_generic(register_tensor, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        return sb.flush_stmts()


class LowerLoadStorePass(Pass):
    def process_function(self, function: Function) -> Function:
        rewriter = LowerLoadStoreRewriter()
        return rewriter.visit(function)


def lower_load_store_pass() -> Pass:
    return LowerLoadStorePass()
