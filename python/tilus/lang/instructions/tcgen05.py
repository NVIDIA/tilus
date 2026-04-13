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
from typing import Optional, Sequence

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.type import DataType
from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor

from .root import InstructionGroup


class Tcgen05InstructionGroup(InstructionGroup):
    """Tensor Core Generation 05 (tcgen05) instructions for Blackwell GPUs.

    Blackwell introduces **tensor memory (TMEM)**, a high-bandwidth on-chip memory space dedicated
    to the tensor core. Unlike registers (which are per-thread), TMEM is a shared accumulator space
    that persists across loop iterations without the cost of register spilling.

    The tcgen05 instruction group manages the full lifecycle of TMEM tensors:

    - **Allocation**: ``alloc()`` / ``dealloc()`` manage TMEM capacity. ``relinquish_alloc_permit()``
      yields allocation rights to a peer CTA when using ``cta_group=2``.
    - **Views**: ``slice()`` and ``view()`` create sub-region or reinterpreted views without copying.
    - **Data movement**: ``load()`` / ``store()`` transfer between TMEM and registers.
      ``copy()`` transfers from shared memory to TMEM. All are async and require
      ``wait_load()`` / ``wait_store()`` or ``commit()`` for synchronization.
    - **Compute**: ``mma()`` performs matrix multiply-accumulate with the accumulator in TMEM,
      supporting both shared-memory and TMEM operands for the A matrix.
    - **Synchronization**: ``commit()`` signals an mbarrier when pending async operations complete.

    With ``cta_group=2``, two CTAs in the same cluster collaborate: each CTA provides half the
    data (split along the M dimension) and holds half the accumulator, enabling larger tile sizes.
    """

    def alloc(self, dtype: DataType, shape: Sequence[int], cta_group: int = 1) -> TMemoryTensor:
        """Allocate a tensor in tensor memory (TMEM).

        Tensor memory is a high-bandwidth on-chip memory space accessible by the tensor core
        on Blackwell GPUs. The allocated tensor can be used as an accumulator for MMA operations
        or for load/store operations.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements (e.g., ``float32``, ``float16``).
        shape: Sequence[int]
            The shape of the tensor. Must have at least 2 dimensions. The second-to-last
            dimension (``shape[-2]``) must be 32, 64, or 128.
        cta_group: int
            The CTA group size for the allocation. Must be 1 or 2. When 2, the tensor is
            shared across two CTAs in the same cluster.

        Returns
        -------
        ret: TMemoryTensor
            The allocated tensor memory tensor.

        Notes
        -----
        - **Thread group**: Must be executed by a thread group with at least 32 threads (one warp).
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.alloc``
        """
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
        """Deallocate a tensor memory tensor.

        Releases the tensor memory previously allocated with ``tcgen05.alloc``.

        Parameters
        ----------
        tensor: TMemoryTensor
            The tensor memory tensor to deallocate.

        Notes
        -----
        - **Thread group**: Must be executed by a thread group with at least 32 threads (one warp).
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.dealloc``
        """
        self._builder.tcgen05_dealloc(tensor)

    def slice(
        self, tensor: TMemoryTensor, offsets: Sequence[Expr | int], dims: Sequence[int], shape: Sequence[int]
    ) -> TMemoryTensor:
        """Create a sliced view of a tensor memory tensor.

        Returns a new ``TMemoryTensor`` that refers to a sub-region of the original tensor.
        This is a metadata-only operation and does not copy data.

        Parameters
        ----------
        tensor: TMemoryTensor
            The source tensor memory tensor.
        offsets: Sequence[Expr | int]
            The starting offsets for each dimension being sliced.
        dims: Sequence[int]
            The dimensions along which to slice.
        shape: Sequence[int]
            The shape of the resulting sliced tensor.

        Returns
        -------
        ret: TMemoryTensor
            A new tensor memory tensor referencing the sliced sub-region.
        """
        return self._builder.tcgen05_slice(tensor, offsets, dims, shape)

    def view(self, tensor: TMemoryTensor, dtype: DataType, shape: Sequence[int]) -> TMemoryTensor:
        """Reinterpret a tensor memory tensor with a different dtype and shape.

        Returns a new ``TMemoryTensor`` that views the same underlying tensor memory with
        a different data type and shape. This is a metadata-only operation.

        Parameters
        ----------
        tensor: TMemoryTensor
            The source tensor memory tensor.
        dtype: DataType
            The new data type for the view.
        shape: Sequence[int]
            The new shape for the view.

        Returns
        -------
        ret: TMemoryTensor
            A new tensor memory tensor with the specified dtype and shape.
        """
        return self._builder.tcgen05_view(tensor, dtype, shape)

    def relinquish_alloc_permit(self, cta_group: int) -> None:
        """Relinquish the tensor memory allocation permit.

        After this instruction, the current CTA group can no longer allocate tensor memory
        until the permit is re-acquired (which happens implicitly after deallocation).
        This allows the peer CTA group to allocate tensor memory.

        Parameters
        ----------
        cta_group: int
            The CTA group whose allocation permit is being relinquished. Must be 1 or 2.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.relinquish_alloc_permit``
        """
        self._builder.tcgen05_relinquish_alloc_permit(cta_group)

    def load(self, tensor: TMemoryTensor) -> RegisterTensor:
        """Load data from tensor memory into registers.

        Copies the contents of a 2D tensor memory tensor into a register tensor.

        Parameters
        ----------
        tensor: TMemoryTensor
            The source tensor memory tensor. Must be 2D.

        Returns
        -------
        ret: RegisterTensor
            A register tensor containing the loaded data.

        Notes
        -----
        - **Thread group**: Must be executed by a warp-aligned thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.ld.sync.aligned``
        """
        if len(tensor.shape) != 2:
            raise InstructionError("load requires a 2D tensor memory tensor, got shape {}".format(tensor.shape))
        return self._builder.tcgen05_load(tensor)

    def store(self, tensor: TMemoryTensor, src: RegisterTensor) -> None:
        """Store data from registers into tensor memory.

        Copies the contents of a register tensor into a 2D tensor memory tensor.

        Parameters
        ----------
        tensor: TMemoryTensor
            The destination tensor memory tensor. Must be 2D.
        src: RegisterTensor
            The source register tensor. Must be 2D.

        Notes
        -----
        - **Thread group**: Must be executed by a warp-aligned thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.st.sync.aligned``
        """
        if len(tensor.shape) != 2:
            raise InstructionError("store requires a 2D tensor memory tensor, got shape {}".format(tensor.shape))
        if len(src.shape) != 2:
            raise InstructionError("store requires a 2D register tensor, got shape {}".format(src.shape))
        return self._builder.tcgen05_store(tensor, src)

    def wait_load(self) -> None:
        """Wait for all pending tensor memory load operations to complete.

        Must be called after ``tcgen05.load`` before using the loaded register data.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.wait::ld``
        """
        self._builder.tcgen05_wait_load()

    def wait_store(self) -> None:
        """Wait for all pending tensor memory store operations to complete.

        Must be called after ``tcgen05.store`` before reading from the destination tensor memory.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.wait::st``
        """
        self._builder.tcgen05_wait_store()

    def copy(self, src: SharedTensor, dst: TMemoryTensor) -> None:
        """Copy data from shared memory to tensor memory.

        Asynchronously copies a 2D shared tensor into a 2D tensor memory tensor. Use
        ``tcgen05.commit`` to signal completion via an mbarrier.

        Parameters
        ----------
        src: SharedTensor
            The source shared tensor. Must be 2D.
        dst: TMemoryTensor
            The destination tensor memory tensor. Must be 2D.

        Notes
        -----
        - **Thread group**: Must be executed by a warp-aligned thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.cp``
        """
        if len(src.shape) != 2:
            raise InstructionError("copy requires a 2D shared tensor, got shape {}".format(src.shape))
        if len(dst.shape) != 2:
            raise InstructionError("copy requires a 2D tensor memory tensor, got shape {}".format(dst.shape))
        self._builder.tcgen05_copy(src, dst)

    def commit(self, mbarrier: Expr | RegisterTensor, cta_group: int = 1, multicast_mask: Optional[int] = None) -> None:
        """Commit pending tcgen05 async operations and signal an mbarrier.

        Groups all prior uncommitted tcgen05 async operations (e.g., ``copy``, ``mma``) and
        signals the specified mbarrier upon completion. The mbarrier's tx-count will be
        decreased when the operations finish.

        Parameters
        ----------
        mbarrier: Expr | RegisterTensor
            The memory barrier to signal upon completion.
        cta_group: int
            The CTA group size. Must be 1 or 2.
        multicast_mask: Optional[int]
            If provided, signals mbarriers on multiple CTAs in the cluster specified by
            the bitmask.

        Notes
        -----
        - **Thread group**: Must be executed by a single warp (use ``self.single_warp()``).
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.commit``
        """
        num_threads = self._builder.tg_stack.current_num_threads
        if num_threads != 32:
            raise InstructionError(
                "tcgen05.commit must be called by a warp (32 threads), got {} threads".format(num_threads)
            )
        self._builder.tcgen05_commit(mbarrier, cta_group, multicast_mask)

    def mma(
        self,
        a: SharedTensor | TMemoryTensor,
        b: SharedTensor,
        d: TMemoryTensor,
        enable_input_d: Expr | bool,
        cta_group: int = 1,
    ) -> None:
        """Perform tensor core matrix multiply-accumulate with TMEM accumulator.

        Computes ``d = a @ b + d`` (or ``d = a @ b`` when ``enable_input_d=False``).
        All operands must be 2D: ``a`` is ``[M, K]``, ``b`` is ``[K, N]``, ``d`` is ``[M, N]``.

        When ``cta_group=2``, two CTAs collaborate on the MMA. Each CTA provides half the
        operands and holds half the accumulator:

        - ``A = [a0; a1]`` — A has shape (M, K), each CTA provides (M/2, K)
        - ``B = [b0, b1]`` — B has shape (K, N), each CTA provides (K, N/2)
        - ``D = [d0; d1]`` — D has shape (M, N), each CTA holds (M/2, N)

        CTA0 is the CTA whose cluster rank has last bit 0, CTA1 is the other.

        Parameters
        ----------
        a: SharedTensor | TMemoryTensor
            Left-hand operand ``[M, K]``. Can be in shared memory or tensor memory.
        b: SharedTensor
            Right-hand operand ``[K, N]``. Must be in shared memory.
        d: TMemoryTensor
            Accumulator ``[M, N]`` in tensor memory. Used as both input and output.
        enable_input_d: Expr | bool
            If ``True``, computes ``d = a @ b + d``. If ``False``, computes ``d = a @ b``.
        cta_group: int
            CTA group size. 1 for single-CTA, 2 for two-CTA collaborative MMA.

        Notes
        -----
        - **Thread group**: Must be executed by a single warp (use ``self.single_warp()``).
        - **Hardware**: Requires compute capability 10.0a+ (sm_100a).
        - **PTX**: ``tcgen05.mma``
        """
        num_threads = self._builder.tg_stack.current_num_threads
        if num_threads != 32:
            raise InstructionError(
                "tcgen05.mma must be called by a warp (32 threads), got {} threads".format(num_threads)
            )
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
