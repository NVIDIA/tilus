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
from __future__ import annotations

from typing import Optional

import numpy as np
from hidet.ir import DataType
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, as_expr, cast, logical_and
from hidet.ir.primitives.cuda.shfl import shfl_down_sync, shfl_up_sync
from hidet.ir.type import tensor_pointer_type
from hidet.utils.py import is_power_of_two

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.instructions.generic import ReduceInst
from tilus.ir.tensor import RegisterTensor
from tilus.ir.utils.veceval import meshgrid, vectorized_evaluate
from tilus.target import nvgpu_any
from tilus.utils import floor_log2, prod


def consume_bits(bits: str, expect: str) -> tuple[str, int]:
    consumed = 0
    while len(bits) > 0 and bits[0] == expect:
        consumed += 1
        bits = bits[1:]
    return bits, consumed


class ReduceScheme:
    """
    Given a number of workers, a ReduceScheme represents a reduction scheme.

    The number of workers must be a power of 2.

    We label the workers from 0 to n-1, where n is the number of workers. The id of each worker can be represented as a
    binary number of m=log2(n) bits: b[m].

    We use a 3-tuple (low, middle, high) to represent the reduction scheme with m = low + middle + high.

    We split the n workers into 2^{lhs + rhs} groups, each group has 2^mid workers. A worker with id b[m] will be in
    group 2^{concat(b[0:low-1], b[low+middle:m-1])}.

    For example, with low=1, middle=3, high=1, we have b = 01001, then we have 4 groups, each group has 8 workers.
    The groups are:
    0: 00000, 00010, 00100, 00110, 01000, 01010, 01100, 01110
    1: 00001, 00011, 00101, 00111, 01001, 01011, 01101, 01111
    2: 10000, 10010, 10100, 10110, 11000, 11010, 11100, 11110
    3: 10001, 10011, 10101, 10111, 11001, 11011, 11101, 11111

    The workers in the same group will be reduced together.
    """

    def __init__(self, low: int, middle: int, high: int):
        self.low: int = low
        self.middle: int = middle
        self.high: int = high
        self.width: int = low + middle + high

    def __str__(self):
        return f"ReduceScheme(low={self.low}, middle={self.middle}, high={self.high})"

    @staticmethod
    def from_worker_array(worker_array: np.ndarray, dim: int, num_workers: int) -> Optional[ReduceScheme]:
        if num_workers == 1:
            return ReduceScheme(low=0, middle=0, high=0)

        shape = worker_array.shape
        """ example (dim=0)
        [[ 0  1  2  3  4  5  6  7]
         [ 0  1  2  3  4  5  6  7]
         [ 8  9 10 11 12 13 14 15]
         [ 8  9 10 11 12 13 14 15]
         [16 17 18 19 20 21 22 23]
         [16 17 18 19 20 21 22 23]
         [24 25 26 27 28 29 30 31]
         [24 25 26 27 28 29 30 31]]
        """

        # normalize the workers to 0 to 2^middle - 1
        worker_array = worker_array - np.min(worker_array, axis=dim, keepdims=True)
        """example
        [[ 0  0  0  0  0  0  0  0]
         [ 0  0  0  0  0  0  0  0]
         [ 8  8  8  8  8  8  8  8]
         [ 8  8  8  8  8  8  8  8]
         [16 16 16 16 16 16 16 16]
         [16 16 16 16 16 16 16 16]
         [24 24 24 24 24 24 24 24]
         [24 24 24 24 24 24 24 24]]
        """

        # check whether each group has the same set of workers
        sorted_worker_array = np.sort(worker_array, axis=dim)
        other_dims = tuple([i for i in range(len(shape)) if i != dim])
        if not np.all((sorted_worker_array - np.min(sorted_worker_array, axis=other_dims, keepdims=True)) == 0):
            return None

        # squeeze other dims, it's safe to use max since all the values in the same group are the same
        worker_array = np.max(sorted_worker_array, axis=other_dims, keepdims=False)
        """example
        [ 0  0  8  8 16 16 24 24]
        """

        # unique the worker array
        worker_array = np.unique(worker_array)
        """example
        [ 0  8 16 24]
        """

        # get the worker with the largest id, it should have the form
        # 00.0 111...111 00.0
        # high    middle  low
        # high: zeros
        # middle: ones
        # low: zeros
        # and then get the size of low, middle, high
        largest_worker = np.max(worker_array)
        nbits = floor_log2(num_workers)
        bits = np.binary_repr(largest_worker, width=nbits)
        bits, high = consume_bits(bits, "0")
        bits, middle = consume_bits(bits, "1")
        bits, low = consume_bits(bits, "0")
        if bits != "":
            return None
        assert high + middle + low == nbits, (high, middle, low, nbits)
        return ReduceScheme(low=low, middle=middle, high=high)


class BlockReduceScheme:
    def __init__(
        self,
        intra_warp_reduce: ReduceScheme,
        inter_warp_reduce: ReduceScheme,
    ):
        self.intra_warp_reduce: ReduceScheme = intra_warp_reduce
        self.inter_warp_reduce: ReduceScheme = inter_warp_reduce

    def __str__(self):
        return (
            f"BlockReduceScheme(intra_warp_reduce={self.intra_warp_reduce}, inter_warp_reduce={self.inter_warp_reduce})"
        )

    @staticmethod
    def from_worker_array(
        worker_array: np.ndarray, dim: int, num_threads: int, warp_size: int = 32
    ) -> Optional[BlockReduceScheme]:
        if not is_power_of_two(num_threads):
            return None
        if num_threads % warp_size != 0:
            return None
        intra_warp_reduce = ReduceScheme.from_worker_array(worker_array % warp_size, dim, num_workers=warp_size)
        inter_warp_reduce = ReduceScheme.from_worker_array(
            worker_array // warp_size, dim, num_workers=num_threads // warp_size
        )
        if intra_warp_reduce is None or inter_warp_reduce is None:
            return None
        return BlockReduceScheme(intra_warp_reduce=intra_warp_reduce, inter_warp_reduce=inter_warp_reduce)


@register_emitter(ReduceInst, target=nvgpu_any)
class ReduceInstEmitter(BaseInstEmitter):
    def analyze_scheme(self, inst: ReduceInst) -> Optional[BlockReduceScheme]:
        src: RegisterTensor = inst.register_input
        dst: RegisterTensor = inst.register_output

        shape = src.shape
        reduced_shape = dst.shape

        assert reduced_shape[inst.dim] == 1
        assert len(shape) == len(reduced_shape)
        assert all(a == b or b == 1 and i == inst.dim for i, (a, b) in enumerate(zip(shape, reduced_shape)))

        global_indices = index_vars(num_vars=len(shape))
        workers = src.layout.get_spatial(global_indices)
        if len(workers) > 1:
            raise ValueError(
                "ReduceInstEmitter only supports single worker reduction currently, but got layout: {}".format(
                    src.layout
                )
            )

        # get the concrete layouts
        global_values: list[np.ndarray] = meshgrid(shape)
        var2values = {a: b for a, b in zip(global_indices, global_values)}
        worker_values: np.ndarray = vectorized_evaluate(as_expr(workers[0]), var2values)

        return BlockReduceScheme.from_worker_array(
            worker_array=worker_values, dim=inst.dim, num_threads=src.layout.spatial_size
        )

    def scalar_init_value(self, op: str, dtype: DataType) -> Expr:
        if op == "sum":
            return dtype.zero
        elif op == "max":
            return dtype.min_value
        elif op == "min":
            return dtype.max_value
        else:
            raise NotImplementedError()

    def scalar_reduce(self, lhs: Expr, rhs: Expr, op: str) -> Expr:
        from hidet.ir.primitives import max, min

        if op == "sum":
            return lhs + rhs
        elif op == "max":
            return max(lhs, rhs)
        elif op == "min":
            return min(lhs, rhs)
        else:
            raise NotImplementedError()

    def intra_thread_reduce(self, src: RegisterTensor, dst: RegisterTensor, inst: ReduceInst) -> None:
        dim = inst.dim
        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        with self.for_range(dst.layout.local_size, attr="u") as dst_local:
            self.buffer_store(dst_buf, indices=[dst_local], value=self.scalar_init_value(inst.op, dst.dtype))
        with self.for_range(src.layout.local_size, attr="u") as src_local:
            global_indices = src.layout.get_global(local_index=src_local, spatial_index=self.current_worker)
            global_indices[dim] = int32.zero
            dst_local = dst.layout.get_local(global_indices=global_indices)
            self.buffer_store(
                dst_buf, indices=[dst_local], value=self.scalar_reduce(dst_buf[dst_local], src_buf[src_local], inst.op)
            )

    def intra_warp_reduce(self, tensor: RegisterTensor, scheme: ReduceScheme, inst: ReduceInst) -> None:
        if scheme.middle == 0:
            # no intra-warp reduction needed
            return

        buf = self.tensor2var[tensor]

        with self.for_range(tensor.layout.local_size, attr="u") as local:
            delta = 1 << (scheme.low + scheme.middle - 1)
            width = 1 << (scheme.low + scheme.middle)

            while delta >= (1 << scheme.low):
                self.buffer_store(
                    buf,
                    indices=[local],
                    value=self.scalar_reduce(
                        lhs=buf[local],
                        rhs=shfl_down_sync(mask=0xFFFFFFFF, var=buf[local], delta=delta, width=width),
                        op=inst.op,
                    ),
                )
                delta >>= 1

    def intra_warp_broadcast(self, tensor: RegisterTensor, scheme: ReduceScheme) -> None:
        if scheme.middle == 0:
            # no intra-warp broadcast needed
            return

        buf = self.tensor2var[tensor]

        with self.for_range(tensor.layout.local_size, attr="u") as local:
            delta = 1 << scheme.low
            width = 1 << (scheme.low + scheme.middle)

            while delta <= (1 << (scheme.low + scheme.middle - 1)):
                self.buffer_store(
                    buf, indices=[local], value=shfl_up_sync(mask=0xFFFFFFFF, var=buf[local], delta=delta, width=width)
                )
                delta <<= 1

    def inter_warp_reduce(
        self, tensor: RegisterTensor, scheme: ReduceScheme, warp_scheme: ReduceScheme, inst: ReduceInst
    ) -> None:
        if scheme.middle == 0:
            # no inter-warp reduction needed
            return

        lane_id = self.declare_var("lane_id", int32, self.current_worker % 32)
        warp_id = self.declare_var("warp_id", int32, self.current_worker // 32)
        regs_buf = self.get_or_allocate_var(tensor)
        smem_buf = self.declare_var(
            "smem_buf",
            tensor_pointer_type(dtype=tensor.dtype, shape=tensor.shape),
            init=cast(self.tensor2var[self.codegen.smem_workspace], ~tensor.dtype),
        )
        middle_mask = ((1 << scheme.middle) - 1) << scheme.low
        lane_middle_mask = ((1 << warp_scheme.middle) - 1) << warp_scheme.low
        with self.for_range(1 << scheme.middle, attr="u") as red_warp_middle:
            with self.if_then(
                logical_and(warp_id & middle_mask == red_warp_middle << scheme.low, lane_id & lane_middle_mask == 0)
            ):
                with self.if_then(red_warp_middle == 0):
                    # store the data from register to shared memory
                    with self.for_range(tensor.layout.local_size) as i:
                        global_indices = tensor.layout.get_global(local_index=i, spatial_index=self.current_worker)
                        self.buffer_store(smem_buf, indices=global_indices, value=regs_buf[i])
                with self.otherwise():
                    # reduce and update shared memory
                    with self.for_range(tensor.layout.local_size) as i:
                        global_indices = tensor.layout.get_global(local_index=i, spatial_index=self.current_worker)
                        self.buffer_store(
                            smem_buf,
                            indices=global_indices,
                            value=self.scalar_reduce(lhs=smem_buf[global_indices], rhs=regs_buf[i], op=inst.op),
                        )
            self.sync()

        # read the data from shared memory
        with self.for_range(tensor.layout.local_size) as i:
            global_indices = tensor.layout.get_global(local_index=i, spatial_index=self.current_worker)
            self.buffer_store(regs_buf, indices=[i], value=smem_buf[global_indices])
        self.sync()

    def efficient_reduce(self, inst: ReduceInst, scheme: BlockReduceScheme) -> None:
        src: RegisterTensor = inst.register_input
        dst: RegisterTensor = inst.register_output

        # print(scheme)

        # reduce within thread
        self.intra_thread_reduce(src, dst, inst)

        # reduce within warp
        self.intra_warp_reduce(dst, scheme.intra_warp_reduce, inst)
        # self.intra_warp_broadcast(dst, scheme.intra_warp_reduce, inst)

        # reduce between warps
        self.inter_warp_reduce(dst, scheme.inter_warp_reduce, scheme.intra_warp_reduce, inst)

        # broadcast within warp
        if scheme.inter_warp_reduce.middle == 0:
            # we only need to do broadcast when we did not do inter-warp reduction
            self.intra_warp_broadcast(dst, scheme.intra_warp_reduce)

    def fallback_reduce(self, inst: ReduceInst) -> None:
        raise NotImplementedError()

    def request_shared_workspace(self, inst: ReduceInst) -> int:
        scheme = self.analyze_scheme(inst)
        if scheme is not None:
            return inst.register_output.dtype.nbytes * prod(inst.register_output.shape)
        else:
            raise NotImplementedError()

    def emit(self, inst: ReduceInst) -> None:
        scheme = self.analyze_scheme(inst)
        if scheme is not None:
            self.efficient_reduce(inst, scheme)
        else:
            self.fallback_reduce(inst)
