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
from typing import Union

from tilus.hidet.ir.expr import Expr
from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor

from .root import InstructionGroup


class WgmmaInstructionGroup(InstructionGroup):
    """Warp Group Matrix Multiply-Accumulate (WGMMA) instructions for Hopper GPUs.

    WGMMA performs asynchronous matrix multiply-accumulate operations using a **warp group**
    (4 consecutive warps, 128 threads). The operands reside in shared memory (``a``, ``b``) or
    registers (``a``), and the accumulator (``d``) is in registers.

    WGMMA operations are asynchronous and must follow a strict execution protocol:

    1. ``fence()`` — establish memory ordering so prior writes to operands are visible.
    2. ``mma()`` — issue one or more async MMA operations (can be called multiple times).
    3. ``commit_group()`` — group all pending MMAs into a commit group.
    4. ``wait_group(n)`` — wait until at most ``n`` commit groups remain pending.

    Multiple commit groups can be in flight simultaneously for latency hiding in pipelined
    loops. For example, issue new MMAs while waiting for a previous group to complete.

    All WGMMA instructions must be executed by a full warp group (4 warps). Use
    ``self.warp_group()`` to create the appropriate thread group context.
    """

    def fence(self) -> None:
        """Issue a warp group MMA fence.

        This fence must be issued before any ``wgmma.mma`` instruction to ensure that prior
        register or shared memory writes are visible to the MMA operation. It establishes
        ordering between generic memory accesses and subsequent wgmma operations.

        Notes
        -----
        - **Thread group**: Must be executed by a warp group (4 warps).
        - **Hardware**: Requires compute capability 9.0a+ (sm_90a).
        - **PTX**: ``wgmma.fence.sync.aligned``
        """
        self._builder.wgmma_fence()

    def commit_group(self) -> None:
        """Commit the previously issued warp group MMA operations.

        Groups all prior uncommitted ``wgmma.mma`` operations into a commit group
        that can be waited on with ``wgmma.wait_group``.

        Notes
        -----
        - **Thread group**: Must be executed by a warp group (4 warps).
        - **Hardware**: Requires compute capability 9.0a+ (sm_90a).
        - **PTX**: ``wgmma.commit_group.sync.aligned``
        """
        self._builder.wgmma_commit_group()

    def wait_group(self, n: Union[Expr, int]) -> None:
        """Wait for warp group MMA commit groups to complete.

        Waits until at most ``n`` commit groups are pending (i.e., all but the most recent
        ``n`` groups have completed).

        Parameters
        ----------
        n: Expr | int
            The number of commit groups allowed to remain pending. Use 0 to wait for all
            committed groups to complete.

        Notes
        -----
        - **Thread group**: Must be executed by a warp group (4 warps).
        - **Hardware**: Requires compute capability 9.0a+ (sm_90a).
        - **PTX**: ``wgmma.wait_group.sync.aligned``
        """
        self._builder.wgmma_wait_group(n)

    def mma(self, a: SharedTensor | RegisterTensor, b: SharedTensor, d: RegisterTensor) -> None:
        """Perform warp group matrix multiply-accumulate (MMA) operation.

        Computes ``d = a @ b + d`` where ``a`` is in shared or register memory, ``b`` is in
        shared memory, and ``d`` is in register memory (both input accumulator and output).

        All tensors must be 2D with compatible shapes: ``a`` is ``[M, K]``, ``b`` is ``[K, N]``,
        and ``d`` is ``[M, N]``.

        A ``wgmma.fence()`` must be called before this instruction, and a ``wgmma.commit_group()``
        followed by ``wgmma.wait_group()`` after.

        Parameters
        ----------
        a: SharedTensor | RegisterTensor
            The left-hand operand of the matrix multiplication. Shape ``[M, K]``.
        b: SharedTensor
            The right-hand operand of the matrix multiplication. Shape ``[K, N]``.
        d: RegisterTensor
            The accumulator tensor, used as both input and output. Shape ``[M, N]``.

        Notes
        -----
        - **Thread group**: Must be executed by a warp group (4 warps).
        - **Hardware**: Requires compute capability 9.0a+ (sm_90a).
        - **PTX**: ``wgmma.mma_async.sync.aligned``
        """
        if any(len(tensor.shape) != 2 for tensor in (a, b, d)):
            raise InstructionError(
                "mma requires 2D tensors, got shapes {}".format([tensor.shape for tensor in (a, b, d)])
            )
        if isinstance(a, SharedTensor):
            self._builder.wgmma_mma_ss(a, b, d)
        elif isinstance(a, RegisterTensor):
            self._builder.wgmma_mma_rs(a, b, d)
        else:
            raise InstructionError("Invalid type of a: {}, expected SharedTensor or RegisterTensor".format(type(a)))
