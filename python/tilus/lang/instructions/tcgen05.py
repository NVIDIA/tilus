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
import contextlib
from typing import Optional, Sequence

from hidet.ir.expr import Expr
from hidet.ir.type import DataType

from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor

from .root import InstructionGroup


class Tcgen05InstructionGroup(InstructionGroup):
    def alloc(
        self, dtype: DataType, shape: Sequence[int], cta_group: int = 1
    ) -> TMemoryTensor:
        if cta_group not in [1, 2]:
            raise InstructionError("cta_group must be 1 or 2")
        if len(shape) < 2:
            raise InstructionError("shape must be a sequence of length 2 or more, got {}".format(shape))
        if shape[-2] not in (32, 64, 128):
            raise InstructionError("shape[-2] must be 32, 64, or 128, got {}".format(shape[-2]))
        if 128 % dtype.nbits != 0:
            raise InstructionError("dtype must be 1, 2, 4, 8, 16, 32, 64, or 128 bit, got {}".format(dtype))
        ret = self._builder.tcgen05_alloc(dtype, shape, cta_group)
        return ret

    def dealloc(self, tensor: TMemoryTensor) -> None:
        self._builder.tcgen05_dealloc(tensor)

    def slice(
        self, tensor: TMemoryTensor, offsets: Sequence[Expr | int], dims: Sequence[int], shape: Sequence[int]
    ) -> TMemoryTensor:
        return self._builder.tcgen05_slice(tensor, offsets, dims, shape)

    def view(self, tensor: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> TMemoryTensor:
        return self._builder.tcgen05_view(tensor, dtype, shape)

    def relinquish_alloc_permit(self, cta_group: int) -> None:
        self._builder.tcgen05_relinquish_alloc_permit(cta_group)

    def load(self, tensor: TMemoryTensor) -> RegisterTensor:
        if len(tensor.shape) != 2:
            raise InstructionError("load requires a 2D tensor memory tensor, got shape {}".format(tensor.shape))
        return self._builder.tcgen05_load(tensor)

    def store(self, tensor: TMemoryTensor, src: RegisterTensor) -> None:
        if len(tensor.shape) != 2:
            raise InstructionError("store requires a 2D tensor memory tensor, got shape {}".format(tensor.shape))
        if len(src.shape) != 2:
            raise InstructionError("store requires a 2D register tensor, got shape {}".format(src.shape))
        return self._builder.tcgen05_store(tensor, src)

    def wait_load(self) -> None:
        self._builder.tcgen05_wait_load()

    def wait_store(self) -> None:
        self._builder.tcgen05_wait_store()

    def copy(self, src: SharedTensor, dst: TMemoryTensor) -> None:
        if len(src.shape) != 2:
            raise InstructionError("copy requires a 2D shared tensor, got shape {}".format(src.shape))
        if len(dst.shape) != 2:
            raise InstructionError("copy requires a 2D tensor memory tensor, got shape {}".format(dst.shape))
        self._builder.tcgen05_copy(src, dst)

    def commit(self, mbarrier: Expr | RegisterTensor, cta_group: int = 1, multicast_mask: Optional[int] = None) -> None:
        if self._builder.tg_stack.current_num_threads != 1:
            raise InstructionError("tcgen05.commit must be called by a single thread")
        self._builder.tcgen05_commit(mbarrier, cta_group, multicast_mask)

    def mma(self, a: SharedTensor | TMemoryTensor, b: SharedTensor, d: TMemoryTensor, enable_input_d: Expr, cta_group: int = 1) -> None:
        """
        Perform tensor core matrix multiply-accumulate (MMA) operation.

        This instruction performs MMA operation: D = A @ B + D, where A, B, and D are matrices.
        The matrices A, B, and D must be 2D tensors. A can be either in shared memory or tensor memory,
        while B must be in shared memory and D must be in tensor memory.

        The `cta_group` parameter specifies which the CTA(s) that will execute the MMA operation.
        - If `cta_group` is 1, the instruction will be executed by the current CTA only.
        - If `cta_group` is 2, the instruction will be executed by the current CTA and its peer CTA in the same cluster.

        The shape of A, B, and D are (M, K), (K, N), and (M, N) respectively.

        When `cta_group` is 1, the given tensors `a`, `b`, and `d` match the shape requirements above directly.

        When `cta_group` is 2, the computation of the MMA is performed collaboratively by two CTAs, and each CTA will provide part of
        the input tensors and hold part of the output tensor. We name the CTA whose CTA rank in the cluster has last bit 0 as CTA0, and the other as CTA1.
        We use `a0`, `b0`, `d0` to denote the slices of A, B, and D provided by CTA0, and 'a1', `b1`, `d1` for CTA1.
        We can represent A, B, and D as follows:
        - A = [a0]
              [a1]
            where A has shape (M, K), a0 and a1 each has shape (M/2, K)
        - B = [b0, b1]
            where B has shape (K, N), b0 and b1 each has shape (K, N/2)
        - D = [d0]
              [d1]
            where D has shape (M, N), d0 and d1 each has shape (M/2, N)
        
        The parameter `enable_input_d` is a boolean expression that indicates whether the input D should be used as the initial value of the accumulator in the MMA operation.
        When `enable_input_d == False`, what this instruction does becomes `D = A @ B`. Otherwise, the instruction performs `D = A @ B + D`.

        Parameters
        ----------
        a: SharedTensor or TMemoryTensor
            The first input matrix. Must be a 2D tensor. Can be in shared memory or tensor memory.
        b: SharedTensor
            The second input matrix. Must be a 2D tensor in shared memory.
        d: TMemoryTensor
            The output matrix. Must be a 2D tensor in tensor memory. It also serves as the accumulator input.
        enable_input_d: Expr
            A boolean expression indicating whether the input D should be used as the initial value of the accumulator in the MMA operation.
        cta_group: int
            The CTA group that executes the MMA operation. Must be either 1 or 2.
        """
        if self._builder.tg_stack.current_num_threads != 1:
            raise InstructionError("tcgen05.mma must be called by a single thread")
        if cta_group not in (1, 2):
            raise InstructionError("cta_group must be 1 or 2, got {}".format(cta_group))
        if isinstance(a, SharedTensor):
            if len(a.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(a.shape))
            if len(b.shape) != 2:
                raise InstructionError("mma requires 2D shared tensors, got shape {}".format(b.shape))
            if len(d.shape) != 2:
                raise InstructionError(
                    "mma requires a 2D tensor memory tensor for output, got shape {}".format(d.shape)
                )
            self._builder.tcgen05_mma_ss(a, b, d, enable_input_d=enable_input_d, cta_group=cta_group)
        elif isinstance(a, TMemoryTensor):
            if len(a.shape) != 2:
                raise InstructionError("mma requires a 2D tensor memory tensor, got shape {}".format(a.shape))
            if len(b.shape) != 2:
                raise InstructionError("mma requires a 2D shared tensor, got shape {}".format(b.shape))
            if len(d.shape) != 2:
                raise InstructionError(
                    "mma requires a 2D tensor memory tensor for output, got shape {}".format(d.shape)
                )
            self._builder.tcgen05_mma_ts(a, b, d, enable_input_d=enable_input_d, cta_group=cta_group)
        else:
            raise InstructionError(f"Invalid type of a: {type(a)}, expected SharedTensor or TMemoryTensor")
