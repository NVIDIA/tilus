from typing import Dict, List, Optional, Sequence

from hidet.ir.dtypes import boolean, int32, uint32
from hidet.ir.expr import Expr, Var, cast, if_then_else
from hidet.ir.primitives.cuda.cp_async import cp_async_commit_group, cp_async_wait_all, cp_async_wait_group
from hidet.ir.type import DataType
from hidet.ir.utils.index_transform import index_deserialize
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.dtypes import uint32x2, uint32x4
from tilus.extensions.hidet.ir.primitives.cuda.cp_async import cp_async
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir.instructions import (
    CopyAsyncCommitGroupInst,
    CopyAsyncGenericInst,
    CopyAsyncWaitAllInst,
    CopyAsyncWaitGroupInst,
)
from tilus.ir.tensor import SharedLayout, SharedTensor
from tilus.target import nvgpu_sm80
from tilus.utils import prod


@register_emitter(CopyAsyncGenericInst, target=nvgpu_sm80)
class CopyAysncInstEmitter(BaseInstEmitter):
    def emit(self, inst: CopyAsyncGenericInst) -> None:
        from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid

        dst: SharedTensor = inst.inputs[0].as_shared_tensor()
        dtype: DataType = dst.dtype
        layout: SharedLayout = dst.layout
        shape: Sequence[int] = layout.shape

        # get shared info
        shared_info: TensorInfo = analyze_grid(shape=layout.shape, axes=layout.axes, var2info={}, expr=layout.offset)

        # get global and mask info
        var2info: Dict[Var, TensorInfo] = {}
        for var, divisibility in self.codegen.function.metadata.analysis.divisibility.items():
            var2info[var] = TensorInfo.from_divisibility(shape=shape, divisibility=divisibility)
        inst_mask = inst.mask if inst.mask is not None else boolean.true
        mask_info: TensorInfo = analyze_grid(shape=shape, axes=inst.axes, var2info=var2info, expr=inst_mask)
        global_info: TensorInfo = analyze_grid(shape=shape, axes=inst.axes, var2info=var2info, expr=inst.offset)

        contiguous_dim: Optional[int] = None
        cp_size: Optional[int] = None
        for nbytes in [16, 8, 4]:
            nbits = nbytes * 8
            for dim in reversed(range(len(shape))):
                if global_info.infos[dim].continuity == 1:
                    continue
                if global_info.infos[dim].divisibility * dtype.nbits % nbits != 0:
                    continue
                if shared_info.infos[dim].continuity * dtype.nbits % nbits != 0:
                    continue
                if shared_info.infos[dim].divisibility * dtype.nbits % nbits != 0:
                    continue
                if mask_info.infos[dim].constancy * dtype.nbits % nbits != 0:
                    continue
                contiguous_dim = dim
                cp_size = nbytes
                break
            if contiguous_dim is not None:
                break

        if contiguous_dim is None:
            attrs = {
                "dtype": str(dtype),
                "shared layout": str(inst.inputs[0].as_shared_tensor().layout),
                "offset": str(inst.offset),
                "mask": str(inst.mask),
                "global_info": str(global_info),
                "shared_info": str(shared_info),
                "mask_info": str(mask_info),
            }
            raise ValueError(
                "The layout/offset/mask is not valid in cp async instruction:\n{}".format(
                    "\n".join(["{}: {}".format(k, v) for k, v in attrs.items()])
                )
            )
        assert cp_size is not None

        cp_dtype = self.get_cp_dtype(cp_size)
        # smem_addr: Var = self.declare(v=Var("smem_addr", int32), init=cvta_generic_to_shared(self.value2var[dst]))
        smem_addr: Var = self.shared_tensor_shared_space_addr[dst]

        if cp_size * 8 % dtype.nbits == 0:
            # a single cp.async instruction copies multiple elements
            # the task (i.e., cp.async instruction) shape
            vec_size = cp_size * 8 // dtype.nbits
            task_shape = [extent if dim != contiguous_dim else extent // vec_size for dim, extent in enumerate(shape)]

            def get_element_indices(task_indices: List[Expr]) -> List[Expr]:
                return [idx if dim != contiguous_dim else idx * vec_size for dim, idx in enumerate(task_indices)]

            def get_global_address_and_mask(task_indices: List[Expr]) -> tuple[Expr, Expr]:
                element_indices = get_element_indices(task_indices)
                remap = {a: b for a, b in zip(inst.axes, element_indices)}
                gmem_offset = rewrite(inst.offset, rewrite_map=remap)
                global_address = cast(inst.ptr, ~dtype) + gmem_offset
                return global_address, rewrite(inst_mask, rewrite_map=remap)

            def get_shared_address(task_indices: List[Expr]) -> Expr:
                element_indices = get_element_indices(task_indices)
                smem_offset = layout(*element_indices)
                return smem_addr + smem_offset * dtype.nbytes

        elif dtype.nbits % cp_size * 8 == 0:
            # a single element needs more cp.async instructions to copy
            # e.g., dtype == uint64x4 which has 256 bits
            vec_size = dtype.nbits // (cp_size * 8)
            task_shape = [extent if idx != contiguous_dim else extent * vec_size for idx, extent in enumerate(shape)]

            def get_element_indices_and_lane_index(task_indices: List[Expr]) -> tuple[List[Expr], Expr]:
                return (
                    # element indices
                    [idx if dim != contiguous_dim else idx // vec_size for dim, idx in enumerate(task_indices)],
                    # lane index
                    task_indices[contiguous_dim] % vec_size,
                )

            def get_global_address_and_mask(task_indices: List[Expr]) -> tuple[Expr, Expr]:
                element_indices, lane_index = get_element_indices_and_lane_index(task_indices)
                remap = {a: b for a, b in zip(inst.axes, element_indices)}
                gmem_offset = rewrite(inst.offset, rewrite_map=remap)
                gmem_offset = gmem_offset * vec_size + lane_index
                global_address = cast(inst.ptr, ~cp_dtype) + gmem_offset
                return global_address, rewrite(inst_mask, rewrite_map=remap)

            def get_shared_address(task_indices: List[Expr]) -> Expr:
                element_indices, lane_index = get_element_indices_and_lane_index(task_indices)
                smem_offset = layout(*element_indices)
                smem_offset = smem_offset * vec_size + lane_index
                return smem_addr + smem_offset * cp_dtype.nbytes

        else:
            raise NotImplementedError()

        def emit_cp_async(task_indices: List[Expr]) -> None:
            global_address, mask = get_global_address_and_mask(task_indices)
            shared_address = get_shared_address(task_indices)

            # gmem_buf: Var = cast(inst.ptr, ~inst.inputs[0].dtype)
            # task_indices: List[Expr] = vec_indices[:-1] + [vec_indices[-1] * vec_size]
            # remap = {a: b for a, b in zip(inst.axes, task_indices)}
            # offset = rewrite(inst.offset, remap)
            # mask = rewrite(mask, remap)
            self.append(
                cp_async(
                    dst=shared_address,
                    src=global_address,
                    cp_size=cp_size,
                    use_shared_space_dst=True,
                    src_size=if_then_else(mask, int32(cp_size), int32.zero),
                    evict=inst.evict,
                    cache_level="global" if cp_size == 16 else "always",  # since `global` requires cp_size=16
                    prefetch_bytes=256,
                )
            )

        num_tasks: int = prod(task_shape)
        num_threads: int = self.num_warps * 32
        if num_tasks < num_threads:
            with self.if_then(self.current_worker < num_tasks):
                emit_cp_async(task_indices=index_deserialize(self.current_worker, shape=task_shape))
        elif num_tasks % num_threads == 0:
            with self.for_range(extent=num_tasks // num_threads, attr="u+") as iter_i:
                emit_cp_async(
                    task_indices=index_deserialize(iter_i * num_threads + self.current_worker, shape=task_shape)
                )
        else:
            with self.for_range(extent=(num_tasks + num_threads - 1) // num_threads, attr="u+") as iter_i:
                with self.if_then(iter_i * num_threads + self.current_worker < num_tasks):
                    emit_cp_async(
                        task_indices=index_deserialize(iter_i * num_threads + self.current_worker, shape=task_shape)
                    )

    @staticmethod
    def get_cp_dtype(cp_size: int) -> DataType:
        """
        Given the cp_size (in bytes), returns a data type with such size.
        """
        return {16: uint32x4, 8: uint32x2, 4: uint32}[cp_size]


@register_emitter(CopyAsyncCommitGroupInst, target=nvgpu_sm80)
class CopyAysncCommitGroupInstEmitter(BaseInstEmitter):
    def emit(self, inst: CopyAsyncCommitGroupInst) -> None:
        self.append(cp_async_commit_group())


@register_emitter(CopyAsyncWaitGroupInst, target=nvgpu_sm80)
class CopyAysncWaitGroupInstEmitter(BaseInstEmitter):
    def emit(self, inst: CopyAsyncWaitGroupInst) -> None:
        self.append(cp_async_wait_group(inst.n))


@register_emitter(CopyAsyncWaitAllInst, target=nvgpu_sm80)
class CopyAysncWaitAllInstEmitter(BaseInstEmitter):
    def emit(self, inst: CopyAsyncWaitAllInst) -> None:
        self.append(cp_async_wait_all())
