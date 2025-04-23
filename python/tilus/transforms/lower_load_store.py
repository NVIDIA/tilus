from typing import Any, Sequence, Union

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from tilus import SharedLayout
from tilus.extensions.hidet.ir.utils.index_transform import index_within_bound
from tilus.ir.builders import StmtBuilder
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import CopyAsyncInst, LoadGlobalInst, LoadSharedInst, StoreGlobalInst, StoreSharedInst
from tilus.ir.layout import GlobalLayout
from tilus.ir.stmt import Stmt
from tilus.transforms.base import Pass


class LowerLoadStoreRewriter(IRRewriter):
    @staticmethod
    def get_funcs(
        offsets: tuple[Expr, ...], dims: tuple[int, ...], layout: GlobalLayout | SharedLayout
    ) -> tuple[Any, Any]:
        def f_global_indices(indices: Sequence[Var]) -> list[Expr]:
            global_indices: list[Expr] = list(offsets)
            for i, dim in enumerate(sorted(dims)):
                global_indices[dim] = global_indices[dim] + indices[i]
            return global_indices

        def f_offset(indices: Sequence[Var]) -> Expr:
            return layout(*f_global_indices(indices))

        def f_mask(indices: Sequence[Var]) -> Expr:
            global_indices = f_global_indices(indices)
            return index_within_bound(global_indices, 0, layout.shape)

        return f_offset, f_mask

    def visit_LoadGlobalInst(self, inst: LoadGlobalInst) -> Stmt:
        inst = super().visit_Instruction(inst)

        sb = StmtBuilder()
        global_tensor = inst.inputs[0].as_global_tensor()
        register_tensor = inst.register_output
        ptr = sb.tensor_ptr(global_tensor)

        f_offset, f_mask = self.get_funcs(offsets=inst.offsets, dims=inst.dims, layout=global_tensor.layout)

        self.memo[inst.register_output] = sb.load_global_generic(
            dtype=global_tensor.dtype, layout=register_tensor.layout, ptr=ptr, f_offset=f_offset, f_mask=f_mask
        )
        return sb.flush_stmts()

    def visit_StoreGlobalInst(self, inst: StoreGlobalInst) -> Stmt:
        inst = super().visit_Instruction(inst)

        sb = StmtBuilder()
        global_tensor = inst.inputs[0].as_global_tensor()
        register_tensor = inst.inputs[1].as_register_tensor()
        ptr = sb.tensor_ptr(global_tensor)

        f_offset, f_mask = self.get_funcs(offsets=inst.offsets, dims=inst.dims, layout=global_tensor.layout)

        sb.store_global_generic(register_tensor, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        return sb.flush_stmts()

    def visit_LoadSharedInst(self, inst: LoadSharedInst) -> Union[Instruction, Stmt]:
        inst = super().visit_Instruction(inst)

        sb = StmtBuilder()
        register_tensor = inst.register_output
        shared_tensor = inst.shared_input
        rank = len(register_tensor.shape)
        offsets = tuple(int32.zero for _ in range(rank))
        dims = tuple(range(rank))
        f_offset, f_mask = self.get_funcs(offsets=offsets, dims=dims, layout=shared_tensor.layout)
        ptr = sb.tensor_ptr(shared_tensor)

        self.memo[inst.register_output] = sb.load_shared_generic(
            dtype=shared_tensor.dtype, layout=register_tensor.layout, ptr=ptr, f_offset=f_offset, f_mask=f_mask
        )
        return sb.flush_stmts()

    def visit_StoreSharedInst(self, inst: StoreSharedInst) -> Union[Instruction, Stmt]:
        inst = super().visit_Instruction(inst)

        sb = StmtBuilder()
        shared_tensor = inst.inputs[0].as_shared_tensor()
        register_tensor = inst.inputs[1].as_register_tensor()
        rank = len(register_tensor.shape)
        offsets = tuple(int32.zero for _ in range(rank))
        dims = tuple(range(rank))
        f_offset, f_mask = self.get_funcs(offsets=offsets, dims=dims, layout=shared_tensor.layout)
        ptr = sb.tensor_ptr(shared_tensor)

        sb.store_shared_generic(register_tensor, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        return sb.flush_stmts()

    def visit_CopyAsyncInst(self, inst: CopyAsyncInst) -> Union[Instruction, Stmt]:
        inst = super().visit_Instruction(inst)

        sb = StmtBuilder()
        shared_tensor = inst.inputs[0].as_shared_tensor()
        global_tensor = inst.inputs[1].as_global_tensor()
        ptr = sb.tensor_ptr(global_tensor)

        dims = tuple(range(len(shared_tensor.shape))) if inst.dims is None else inst.dims

        f_offset, f_mask = self.get_funcs(offsets=inst.offsets, dims=dims, layout=global_tensor.layout)

        sb.copy_async_generic(
            dst=shared_tensor, ptr=ptr, f_offset=f_offset, f_mask=f_mask, evict=inst.evict, weak_mask=inst.weak_mask
        )
        return sb.flush_stmts()


class LowerLoadStorePass(Pass):
    def process_function(self, function: Function) -> Function:
        rewriter = LowerLoadStoreRewriter()
        return rewriter.visit(function)


def lower_load_store_pass() -> Pass:
    return LowerLoadStorePass()
