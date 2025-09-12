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
from typing import List, Optional, Sequence

from hidet.ir import logical_and
from hidet.ir.dtypes import boolean, int32, uint32, uint8
from hidet.ir.expr import Expr, Var
from hidet.ir.primitives import printf
from hidet.ir.primitives.cuda.barrier import fence_view_async_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.utils.index_transform import index_deserialize
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

from tilus.backends.codegen import register_emitter, Codegen
from tilus.backends.emitters.cuda.cp_async_base import CopyAsyncAnalysisResult, CopyAsyncAnalysisSharedToSharedResult, CopyAysncBaseEmitter
from tilus.extensions.hidet.ir.primitives.cuda.copy_async_bulk import cp_async_bulk_global_to_shared, cp_async_bulk_shared_to_global, cp_async_bulk_global_to_cluster_shared, cp_async_bulk_shared_to_cluster_shared
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import mbarrier_expect_tx_cta_shared, mbarrier_expect_tx_cluster_shared
from tilus.ir import GlobalTensor
from tilus.ir.instructions import CopyAsyncBulkGlobalToSharedInst, CopyAsyncBulkSharedToGlobalInst, CopyAsyncBulkSharedToClusterSharedInst, CopyAsyncBulkGlobalToClusterSharedInst
from tilus.ir.tensor import SharedTensor
from tilus.target import nvgpu_sm90
from tilus.utils import prod


def within_bound(indices: Sequence[Expr | int], shape: Sequence[Expr | int]) -> Expr:
    assert len(indices) == len(shape)
    conditions = []
    for idx, extent in zip(indices, shape):
        conditions.append(logical_and(0 <= idx, idx < extent))
    return logical_and(*conditions)


class BulkCopyAsyncBetweenGlobalShared(CopyAysncBaseEmitter):
    def emit_cp_async(
        self,
        shared_address: Expr,
        global_address: Expr,
        size: int,
        inst: CopyAsyncBulkSharedToGlobalInst | CopyAsyncBulkGlobalToSharedInst | CopyAsyncBulkGlobalToClusterSharedInst,
    ) -> None:
        raise NotImplementedError()

    def common_emit(
        self,
        shared_tensor: SharedTensor,
        global_tensor: GlobalTensor,
        offsets: Sequence[Expr],
        dims: Optional[Sequence[int]],
        check_bounds: bool,
        inst: CopyAsyncBulkSharedToGlobalInst | CopyAsyncBulkGlobalToSharedInst | CopyAsyncBulkGlobalToClusterSharedInst,
    ) -> None:
        analysis: CopyAsyncAnalysisResult = self.analyze(
            shared_tensor=shared_tensor,
            global_tensor=global_tensor,
            offsets=offsets,
            dims=dims,
            check_bounds=check_bounds,
        )

        if analysis.cp_size_bits % 128 != 0:  # cp.async.bulk only supports 128-bit or larger cp size
            raise ValueError(
                "cp.async.bulk only supports 128-bit or larger cp size. Got the following analysis result: \n{}".format(
                    analysis
                )
            )

        dtype = shared_tensor.dtype
        cp_size = analysis.cp_size_bits // 8
        contiguous_dim = analysis.contiguous_dim

        elements_per_copy = analysis.cp_size_bits // dtype.nbits
        copy_shape = list(shared_tensor.shape)
        assert copy_shape[contiguous_dim] % elements_per_copy == 0
        copy_shape[contiguous_dim] //= elements_per_copy
        num_copies = prod(copy_shape)
        num_threads = self.current_num_workers

        def emit_bulk_cp_async(copy_indices: List[Expr]) -> None:
            shared_indices = list(copy_indices)
            shared_indices[contiguous_dim] = shared_indices[contiguous_dim] * elements_per_copy
            global_indices = list(offsets)
            for i, offset in enumerate(shared_indices):
                if i in dims:
                    global_indices[i] = global_indices[i] + offset

            global_address = self.tensor2var[global_tensor] + global_tensor.layout(*global_indices)
            shared_address = (
                self.shared_tensor_shared_space_addr[shared_tensor]
                + shared_tensor.layout(*shared_indices) * dtype.nbytes
            )
            mask = boolean.true if not check_bounds else within_bound(global_indices, global_tensor.shape)
            with self.if_then(mask):
                self.emit_cp_async(
                    shared_address=shared_address,
                    global_address=global_address,
                    size=cp_size,
                    inst=inst,
                )

        if num_copies % num_threads == 0:
            with self.for_range(extent=num_copies // num_threads, attr="u+") as iter_i:
                emit_bulk_cp_async(index_deserialize(self.current_worker + iter_i * num_threads, shape=copy_shape))
        elif num_copies < num_copies:
            with self.if_then(self.current_worker < num_copies):
                emit_bulk_cp_async(index_deserialize(self.current_worker, shape=copy_shape))
        else:
            with self.for_range(extent=(num_copies + num_threads - 1) // num_threads, attr="u+") as iter_i:
                with self.if_then(self.current_worker + iter_i * num_threads < num_copies):
                    emit_bulk_cp_async(index_deserialize(self.current_worker + iter_i * num_threads, shape=copy_shape))


@register_emitter(CopyAsyncBulkGlobalToSharedInst, target=nvgpu_sm90)
class BulkCopyAsyncGlobalToSharedInstEmitter(BulkCopyAsyncBetweenGlobalShared):
    def __init__(self, codegen: Codegen):
        super().__init__(codegen)
        self.barrier_addr: Optional[Var] = None

    def emit_cp_async(
        self, shared_address: Expr, global_address: Expr, size: int, inst: CopyAsyncBulkGlobalToSharedInst
    ) -> None:
        self.append(mbarrier_expect_tx_cta_shared(mbarrier_addr=self.barrier_addr, transaction_bytes=size))
        self.append(cp_async_bulk_global_to_shared(
            dst=shared_address,
            src=global_address,
            size=int32(size),
            mbarrier=self.barrier_addr,
            l2_evict=inst.evict,
        ))

    def emit(self, inst: CopyAsyncBulkGlobalToSharedInst) -> None:
        self.barrier_addr = self.declare_var(name='barrier_addr', tp=uint32, init=cvta_generic_to_shared(inst.mbarrier))
        self.common_emit(
            shared_tensor=inst.inputs[0].as_shared_tensor(),
            global_tensor=inst.inputs[1].as_global_tensor(),
            offsets=inst.offsets,
            dims=inst.dims,
            check_bounds=inst.check_bounds,
            inst=inst,
        )


@register_emitter(CopyAsyncBulkSharedToGlobalInst, target=nvgpu_sm90)
class CopyAysncBulkSharedToGlobalInstEmitter(BulkCopyAsyncBetweenGlobalShared):
    def emit_cp_async(
        self, shared_address: Expr, global_address: Expr, size: int, inst: CopyAsyncBulkSharedToGlobalInst
    ) -> None:
        self.append(cp_async_bulk_shared_to_global(
            dst=global_address,
            src=shared_address,
            size=int32(size),
            l2_evict=inst.l2_evict,
        ))

    def emit(self, inst: CopyAsyncBulkSharedToGlobalInst) -> None:
        self.append(fence_view_async_shared())
        self.common_emit(
            shared_tensor=inst.inputs[1].as_shared_tensor(),
            global_tensor=inst.inputs[0].as_global_tensor(),
            offsets=inst.offsets,
            dims=inst.dims,
            check_bounds=inst.check_bounds,
            inst=inst,
        )


@register_emitter(CopyAsyncBulkGlobalToClusterSharedInst, target=nvgpu_sm90)
class CopyAsyncBulkGlobalToClusterSharedEmitter(BulkCopyAsyncBetweenGlobalShared):
    def __init__(self, codegen: Codegen):
        super().__init__(codegen)
        self.barrier_addr: Optional[Var] = None

    @staticmethod
    def get_smallest_block_rank(cta_mask: int) -> int:
        for i in range(16):
            if (1 << i) & cta_mask:
                return i
        raise ValueError("Invalid cta_mask: {}".format(cta_mask))


    def emit_cp_async(
        self, shared_address: Expr, global_address: Expr, size: int, inst: CopyAsyncBulkGlobalToClusterSharedInst
    ) -> None:
        with self.if_then((1 << self.block_rank_in_cluster) & inst.cta_mask):
            # self.append(printf(
            #     "[%d, %d, %d][%d][%d] mbarrier_expect_tx_shared: mbarrier=%d (%p) %d, transaction_bytes=%d, sBase64=%p (%d)\n",
            #     self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster, self.current_worker,
            #     barrier_addr, inst.mbarrier, cvta_generic_to_shared(inst.mbarrier), int32(size), dynamic_shared_memory(0, dtype=uint8), cvta_generic_to_shared(dynamic_shared_memory(0, dtype=uint8))
            # ))
            self.append(mbarrier_expect_tx_cta_shared(mbarrier_addr=self.barrier_addr, transaction_bytes=size))
        with self.if_then(self.block_rank_in_cluster == uint32(self.get_smallest_block_rank(inst.cta_mask))):
            # self.append(printf(
            #     "[%d, %d, %d][%d][%d] cp.async.bulk.global.to.cluster.shared: dst=%d, src=%p, size=%d, mbarrier=%d, cta_mask=0x%x\n",
            #     self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster, self.current_worker,
            #     shared_address, global_address, int32(size), barrier_addr, int32(inst.cta_mask)
            # ))
            self.append(cp_async_bulk_global_to_cluster_shared(
                dst=shared_address,
                src=global_address,
                size=int32(size),
                mbarrier=self.barrier_addr,
                cta_mask=inst.cta_mask,
                l2_evict=inst.evict,
            ))

    def emit(self, inst: CopyAsyncBulkGlobalToClusterSharedInst) -> None:
        self.barrier_addr = self.declare_var(name='barrier_addr', tp=uint32, init=cvta_generic_to_shared(inst.mbarrier))
        self.common_emit(
            shared_tensor=inst.inputs[0].as_shared_tensor(),
            global_tensor=inst.inputs[1].as_global_tensor(),
            offsets=inst.offsets,
            dims=inst.dims,
            check_bounds=inst.check_bounds,
            inst=inst,
        )

@register_emitter(CopyAsyncBulkSharedToClusterSharedInst, target=nvgpu_sm90)
class CopyAsyncBulkSharedToClusterSharedEmitter(CopyAysncBaseEmitter):
    def emit(self, inst: CopyAsyncBulkSharedToClusterSharedInst) -> None:
        shared_src = inst.inputs[1].as_shared_tensor()
        shared_dst = inst.inputs[0].as_shared_tensor()
        analysis: CopyAsyncAnalysisSharedToSharedResult = self.analyze_shared_to_shared(
            shared_src=shared_src,
            shared_dst=shared_dst
        )

        if analysis.cp_size_bits % 128 != 0:  # cp.async.bulk only supports 128-bit or larger cp size
            raise ValueError(
                "cp.async.bulk only supports 128-bit or larger cp size. Got the following analysis result: \n{}".format(
                    analysis
                )
            )

        dtype = analysis.dtype
        cp_size = analysis.cp_size_bits // 8
        contiguous_dim = analysis.contiguous_dim

        elements_per_copy = analysis.cp_size_bits // dtype.nbits
        copy_shape = list(shared_dst.shape)
        assert copy_shape[contiguous_dim] % elements_per_copy == 0
        copy_shape[contiguous_dim] //= elements_per_copy
        num_copies = prod(copy_shape)
        num_threads = self.current_num_workers

        barrier_addr = self.declare_var(name='barrier_addr', tp=uint32, init=cvta_generic_to_shared(inst.mbarrier))

        def emit_bulk_cp_async(copy_indices: List[Expr]) -> None:
            shared_indices = list(copy_indices)
            shared_indices[contiguous_dim] = shared_indices[contiguous_dim] * elements_per_copy
            src_addr = self.declare_var(
                'src_addr',
                tp=uint32,
                init=self.shared_tensor_shared_space_addr[shared_src] + shared_src.layout(*shared_indices) * dtype.nbytes
            )
            dst_addr = self.declare_var(
                'dst_addr',
                tp=uint32,
                init=self.shared_tensor_shared_space_addr[shared_dst] + shared_dst.layout(*shared_indices) * dtype.nbytes
            )
            self.append(
                cp_async_bulk_shared_to_cluster_shared(
                    dst=dst_addr,
                    src=src_addr,
                    size=int32(cp_size),
                    mbarrier=barrier_addr
                )
            )

        if num_copies % num_threads == 0:
            with self.for_range(extent=num_copies // num_threads, attr="u+") as iter_i:
                emit_bulk_cp_async(index_deserialize(self.current_worker + iter_i * num_threads, shape=copy_shape))
        elif num_copies < num_copies:
            with self.if_then(self.current_worker < num_copies):
                emit_bulk_cp_async(index_deserialize(self.current_worker, shape=copy_shape))
        else:
            with self.for_range(extent=(num_copies + num_threads - 1) // num_threads, attr="u+") as iter_i:
                with self.if_then(self.current_worker + iter_i * num_threads < num_copies):
                    emit_bulk_cp_async(index_deserialize(self.current_worker + iter_i * num_threads, shape=copy_shape))
