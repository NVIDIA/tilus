# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from typing import Optional, Sequence

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.type import DataType
from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor, TMemoryTensor

from .root import InstructionGroup

# Allowed user-facing names for the tcgen05.copy multicast kwarg. These match
# the duplication-mode names on TMemoryLayout. Stored as plain strings on the
# IR (no enum) so functors walk the field generically; converted to
# Tcgen05CopyMulticastKind at codegen time.
_VALID_COPY_MULTICAST_NAMES: tuple[str, ...] = ("warpx4", "warpx2_02_13", "warpx2_01_23")


# ----------------------------------------------------------------------------
# Block-scaled MMA dispatch table.
#
# Each row maps (operand dtypes + sf_block_size) → the unique PTX
# (kind, scale_vec) and the per-CTA constraints on (M, N, K). Used by
# Tcgen05InstructionGroup.mma_scaled() to validate inputs and select the
# correct PTX instruction.
#
# The keys use dtype short names (e.g. ``"f4e2m1"``, ``"f8e4m3"``,
# ``"f8e8m0"``) so the table doesn't depend on dtype objects existing in
# tilus — paths whose SF dtype isn't yet registered (e.g. UE8M0) will be
# rejected by the dtype-name match before reaching codegen.
# ----------------------------------------------------------------------------

_F8_F6_F4_DTYPES: frozenset[str] = frozenset(
    {"f8e4m3", "f8e5m2", "f6e2m3", "f6e3m2", "f4e2m1"}
)


@dataclass(frozen=True)
class _BlockScaledRow:
    """One row of the block-scaled MMA support matrix.

    Each row represents a unique combination of operand types and SF block
    size, mapping to a single PTX (kind, scale_vec) pair. The row also
    records the legal per-CTA shape envelope: ``M ∈ M_set``,
    ``N ∈ N_set`` (or ``N`` ranges as ``N_min..N_max step N_step``),
    ``K = inst_K``.
    """

    a_dtypes: frozenset[str]  # accepted dtypes for operand A
    b_dtypes: frozenset[str]  # accepted dtypes for operand B
    sf_dtypes: frozenset[str]  # accepted dtypes for SFA / SFB (must match each other)
    sf_block_size: int  # 16 or 32
    cta_group: int  # 1 or 2
    kind: str  # PTX kind: "mxf4nvf4", "mxf4", "mxf8f6f4"
    scale_vec: str  # "1X", "2X", "4X"
    inst_K: int  # per-CTA K dim of A / B
    M_set: tuple[int, ...]  # legal per-CTA M
    N_min: int  # legal per-CTA N: in [N_min, N_max] step N_step
    N_max: int
    N_step: int
    valid_sf_ids: tuple[int, ...]  # legal sfa_id / sfb_id values


# Per-CTA M values: for cta_group=1 only 128 (PTX inst_M = 128). For
# cta_group=2 the inst_M is 128 or 256, so per-CTA M = inst_M / 2 ∈ {64, 128}.
_M_CTA1: tuple[int, ...] = (128,)
_M_CTA2: tuple[int, ...] = (64, 128)


_BLOCK_SCALED_DISPATCH: tuple[_BlockScaledRow, ...] = (
    # ---- mxf4nvf4 + scale_vec::4X (= block16) -------------------------------
    # NVFP4 with UE4M3 SF, K=64
    _BlockScaledRow(
        a_dtypes=frozenset({"f4e2m1"}),
        b_dtypes=frozenset({"f4e2m1"}),
        sf_dtypes=frozenset({"f8e4m3", "f8e8m0"}),
        sf_block_size=16,
        cta_group=1,
        kind="mxf4nvf4",
        scale_vec="4X",
        inst_K=64,
        M_set=_M_CTA1,
        N_min=8, N_max=256, N_step=8,
        valid_sf_ids=(0,),
    ),
    _BlockScaledRow(
        a_dtypes=frozenset({"f4e2m1"}),
        b_dtypes=frozenset({"f4e2m1"}),
        sf_dtypes=frozenset({"f8e4m3", "f8e8m0"}),
        sf_block_size=16,
        cta_group=2,
        kind="mxf4nvf4",
        scale_vec="4X",
        inst_K=64,
        M_set=_M_CTA2,
        N_min=8, N_max=128, N_step=8,
        valid_sf_ids=(0,),
    ),
    # ---- mxf4nvf4 + scale_vec::2X (= block32) -------------------------------
    # MXFP4 (FP4 operands, UE8M0 SF, K=64, block_size=32 → 2 SFs per row per inst).
    _BlockScaledRow(
        a_dtypes=frozenset({"f4e2m1"}),
        b_dtypes=frozenset({"f4e2m1"}),
        sf_dtypes=frozenset({"f8e8m0"}),
        sf_block_size=32,
        cta_group=1,
        kind="mxf4nvf4",
        scale_vec="2X",
        inst_K=64,
        M_set=_M_CTA1,
        N_min=8, N_max=256, N_step=8,
        valid_sf_ids=(0, 2),
    ),
    _BlockScaledRow(
        a_dtypes=frozenset({"f4e2m1"}),
        b_dtypes=frozenset({"f4e2m1"}),
        sf_dtypes=frozenset({"f8e8m0"}),
        sf_block_size=32,
        cta_group=2,
        kind="mxf4nvf4",
        scale_vec="2X",
        inst_K=64,
        M_set=_M_CTA2,
        N_min=8, N_max=128, N_step=8,
        valid_sf_ids=(0, 2),
    ),
    # ---- mxf8f6f4 + scale_vec::1X (= block32) -------------------------------
    # MXFP8 / MXFP6 / MXFP4-mixed (any FP8/FP6/FP4 operands, UE8M0 SF, K=32,
    # block_size=32 → 1 SF per row per inst, 4 K-iters share a TMEM cell via
    # SFA_ID rotation).
    _BlockScaledRow(
        a_dtypes=_F8_F6_F4_DTYPES,
        b_dtypes=_F8_F6_F4_DTYPES,
        sf_dtypes=frozenset({"f8e8m0"}),
        sf_block_size=32,
        cta_group=1,
        kind="mxf8f6f4",
        scale_vec="1X",
        inst_K=32,
        M_set=_M_CTA1,
        N_min=8, N_max=256, N_step=8,
        valid_sf_ids=(0, 1, 2, 3),
    ),
    _BlockScaledRow(
        a_dtypes=_F8_F6_F4_DTYPES,
        b_dtypes=_F8_F6_F4_DTYPES,
        sf_dtypes=frozenset({"f8e8m0"}),
        sf_block_size=32,
        cta_group=2,
        kind="mxf8f6f4",
        scale_vec="1X",
        inst_K=32,
        M_set=_M_CTA2,
        N_min=8, N_max=128, N_step=8,
        valid_sf_ids=(0, 1, 2, 3),
    ),
)


def _lookup_block_scaled_row(
    a_dtype: DataType,
    b_dtype: DataType,
    sfa_dtype: DataType,
    sfb_dtype: DataType,
    sf_block_size: int,
    cta_group: int,
) -> Optional["_BlockScaledRow"]:
    """Return the matching row of the block-scaled dispatch table or ``None``."""
    if sfa_dtype.short_name != sfb_dtype.short_name:
        return None
    for row in _BLOCK_SCALED_DISPATCH:
        if (
            a_dtype.short_name in row.a_dtypes
            and b_dtype.short_name in row.b_dtypes
            and sfa_dtype.short_name in row.sf_dtypes
            and sf_block_size == row.sf_block_size
            and cta_group == row.cta_group
        ):
            return row
    return None


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
            The shape of the tensor. Must have at least 2 dimensions. The first
            dimension (``shape[0]``) is the lane axis and must be 32, 64, or 128.
            All remaining dimensions are column-strided.
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
        if shape[0] not in (32, 64, 128):
            raise InstructionError("shape[0] must be 32, 64, or 128, got {}".format(shape[0]))
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

    def copy(self, src: SharedTensor, dst: TMemoryTensor, multicast: Optional[str] = None) -> None:
        """Copy data from shared memory to tensor memory.

        Asynchronously copies a shared tensor into a tensor memory tensor. Use
        ``tcgen05.commit`` to signal completion via an mbarrier.

        Parameters
        ----------
        src: SharedTensor
            The source shared tensor.
        dst: TMemoryTensor
            The destination tensor memory tensor.
        multicast: Optional[str]
            Multicast pattern for replicating ``src`` across TMEM sub-partitions.

            - ``None`` (default): plain 1:1 copy (no replication).
            - ``"warpx4"``: replicate ``src`` to all 4 warp-aligned 32-lane stripes
              of TMEM. Source ``src`` has 32 unique lane rows; ``dst`` is a TMEM
              tensor with ``WARPX4`` duplication (``shape[0] == 32``).
            - ``"warpx2_02_13"`` / ``"warpx2_01_23"``: replicate ``src`` to two
              warp-pairs (by parity / by halves). Source has 64 unique lane
              rows; ``dst`` has the matching ``WARPX2_*`` duplication
              (``shape[0] == 64``).

        Notes
        -----
        - **Thread group**: Must be executed by a warp-aligned thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``tcgen05.cp[.warpx4 / .warpx2_02_13 / .warpx2_01_23]``
        """
        if multicast is not None and multicast not in _VALID_COPY_MULTICAST_NAMES:
            raise InstructionError(
                "Unknown multicast mode {!r}. Expected None or one of: {}".format(
                    multicast, ", ".join(repr(k) for k in _VALID_COPY_MULTICAST_NAMES)
                )
            )
        self._builder.tcgen05_copy(src, dst, multicast=multicast or "")

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

    def mma_scaled(
        self,
        a: SharedTensor | TMemoryTensor,
        b: SharedTensor,
        d: TMemoryTensor,
        sfa: TMemoryTensor,
        sfb: TMemoryTensor,
        sf_block_size: int,
        enable_input_d: Expr | bool,
        cta_group: int = 1,
        sfa_id: int = 0,
        sfb_id: int = 0,
    ) -> None:
        """Block-scaled tensor-core MMA: ``D = (A * scale_A) * (B * scale_B) + D``.

        Wraps **one** PTX ``tcgen05.mma.cta_group::C.kind::K.block_scale.scale_vec::S``
        instruction. The combination ``(a.dtype, b.dtype, sfa.dtype, sf_block_size,
        cta_group)`` uniquely determines the PTX ``kind`` and ``scale_vec`` modifiers
        via the support matrix below — the user just passes operands and the
        codegen emits a single, fully-resolved PTX call.

        Each kernel-level call corresponds to **exactly one** PTX inst. There is no
        compiler-side splitting along K, no SFA_ID rotation loop, and no inst-shape
        search. If the user wants to amortise multiple inst-K iters into one
        cell-round (the mega-MoE / DeepGEMM trick that uses ``SFA_ID`` ∈ {0..3} to
        time-multiplex 4 K-iters per TMEM cell), they write that loop **at the
        program level** by calling ``mma_scaled`` multiple times with rotating
        ``sfa_id`` / ``sfb_id``.

        Support matrix
        --------------

        Each row maps a unique combination of ``(a/b dtype, sf dtype, sf_block_size,
        cta_group)`` to one PTX ``(kind, scale_vec)`` pair. **Per-CTA shapes**
        (sizes the user passes for ``a``, ``b``, ``d``, ``sfa``, ``sfb``) are listed
        as functions of inst_M / inst_N (the per-CTA M and N of A/B); for cta_group=2
        the partitioning across the CTA pair is documented in the next section.

        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+
        | Row | sf_blk  | a / b dtype       | sf dtype | cta  | M   | N   | K   | a      | b      | d      | sfa      | sfb      | PTX kind     | scale    | sfa_id/  |
        |     | _size   |                   |          | grp  |     |     |     | shape  | shape  | shape  | shape    | shape    |              | _vec     | sfb_id   |
        +=====+=========+===================+==========+======+=====+=====+=====+========+========+========+==========+==========+==============+==========+==========+
        | 1   | 16      | f4e2m1 ²          | f8e4m3   | 1    | 128 | 8.. | 64  | (M, K) | (K, N) | (M, N) | [32,4,4] | [32,N/   | mxf4nvf4     | 4X       | {0}      |
        |     |         |                   | or       |      |     | 256 |     |        |        |        |          | 32, 4]   |              | (block16)|          |
        |     |         |                   | f8e8m0   |      |     | /8  |     |        |        |        |          |          |              |          |          |
        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+
        | 2   | 16      | f4e2m1 ²          | f8e4m3   | 2    | {64,| 8.. | 64  | (M, K) | (K, N) | (M, 2N)| [32,M/   | [32,N/   | mxf4nvf4     | 4X       | {0}      |
        |     |         |                   | or       |      | 128}| 128 |     |        |        |        | 32, 4]   | 32, 4]   |              |          |          |
        |     |         |                   | f8e8m0   |      |     | /8  |     |        |        |        |          |          |              |          |          |
        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+
        | 3   | 32      | f4e2m1 ²          | f8e8m0   | 1    | 128 | 8.. | 64  | (M, K) | (K, N) | (M, N) | [32,4,4] | [32,N/   | mxf4nvf4     | 2X       | {0, 2}   |
        |     |         |                   |          |      |     | 256 |     |        |        |        |          | 32, 4]   |              | (block32)|          |
        |     |         |                   |          |      |     | /8  |     |        |        |        |          |          |              |          |          |
        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+
        | 4   | 32      | f4e2m1 ²          | f8e8m0   | 2    | {64,| 8.. | 64  | (M, K) | (K, N) | (M, 2N)| [32,M/   | [32,N/   | mxf4nvf4     | 2X       | {0, 2}   |
        |     |         |                   |          |      | 128}| 128 |     |        |        |        | 32, 4]   | 32, 4]   |              |          |          |
        |     |         |                   |          |      |     | /8  |     |        |        |        |          |          |              |          |          |
        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+
        | 5   | 32      | any of            | f8e8m0   | 1    | 128 | 8.. | 32  | (M, K) | (K, N) | (M, N) | [32,4,4] | [32,N/   | mxf8f6f4     | 1X       | {0,1,2,3}|
        |     |         | {f4e2m1, f6e2m3,  |          |      |     | 256 |     |        |        |        |          | 32, 4]   |              | (block32)|          |
        |     |         |  f6e3m2, f8e4m3,  |          |      |     | /8  |     |        |        |        |          |          |              |          |          |
        |     |         |  f8e5m2} ²        |          |      |     |     |     |        |        |        |          |          |              |          |          |
        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+
        | 6   | 32      | any of            | f8e8m0   | 2    | {64,| 8.. | 32  | (M, K) | (K, N) | (M, 2N)| [32,M/   | [32,N/   | mxf8f6f4     | 1X       | {0,1,2,3}|
        |     |         | {f4e2m1, f6e2m3,  |          |      | 128}| 128 |     |        |        |        | 32, 4]   | 32, 4]   |              |          |          |
        |     |         |  f6e3m2, f8e4m3,  |          |      |     | /8  |     |        |        |        |          |          |              |          |          |
        |     |         |  f8e5m2} ²        |          |      |     |     |     |        |        |        |          |          |              |          |          |
        +-----+---------+-------------------+----------+------+-----+-----+-----+--------+--------+--------+----------+----------+--------------+----------+----------+

        ``M`` and ``N`` in this table are **per-CTA**; ``a/b/d/sfa/sfb`` shapes are
        what the user passes for one CTA. The "/8" shorthand means *step 8* (so for
        cta_group=1, ``N ∈ {8, 16, 24, …, 256}``). ``f8e8m0`` (UE8M0) requires the
        corresponding tilus dtype to be registered; if not, only ``f8e4m3`` (UE4M3)
        SF rows are usable (the NVFP4 case).

        Partitioning for ``cta_group=2``
        --------------------------------

        The PTX inst is performed by a 2-CTA cluster pair (M-direction split). Each
        CTA passes its own slice of the operands; the MMA hardware reads peer-CTA
        shared memory to recombine the result. The split is:

        +----------+--------+-------------------------------------------+--------------------------+
        | Operand  | Memory | Split axis                                | Per-CTA shape            |
        +==========+========+===========================================+==========================+
        | ``a``    | SMEM   | M (each CTA holds half the M-rows)        | ``(M, K)``               |
        |          | TMEM   |                                           |                          |
        +----------+--------+-------------------------------------------+--------------------------+
        | ``b``    | SMEM   | N (each CTA holds half the N-cols)        | ``(K, N)``               |
        +----------+--------+-------------------------------------------+--------------------------+
        | ``d``    | TMEM   | M only (each CTA holds half-M × full-N)   | ``(M, 2N)``              |
        +----------+--------+-------------------------------------------+--------------------------+
        | ``sfa``  | TMEM   | M (matches A)                             | ``[32, M/32, 4]``        |
        +----------+--------+-------------------------------------------+--------------------------+
        | ``sfb``  | TMEM   | N (matches B)                             | ``[32, N/32, 4]``        |
        +----------+--------+-------------------------------------------+--------------------------+

        ``D`` is intentionally asymmetric — its M-dim is split (matches A) but its
        N-dim is **full** per CTA (PTX layout B/A from §9.7.16.10.5 — each CTA holds
        ``(M/2, N)`` of the cluster's logical (M, N) D). This matches the existing
        ``mma()`` validation: ``d.shape[1] == 2 * b.shape[1]`` for ``cta_group=2``.

        SF tensor layout
        ----------------

        Both ``sfa`` and ``sfb`` must be allocated as TMEM tensors of shape
        ``[32, MN_fold, 4]`` (with ``WARPX4`` duplication, inferred from
        ``shape[0] == 32``):

        - dim 0 = 32 → lane axis (replicated across all 4 sub-partitions per
          PTX's "duplicated to all 32 lane partitions" rule).
        - dim 1 = ``M/32`` for SFA / ``N/32`` for SFB → the M-fold / N-fold
          inside the per-CTA slice.
        - dim 2 = 4 → the 4 byte slots inside each 32-bit TMEM cell. How many
          of those bytes are read **by this single inst** depends on
          ``scale_vec`` (and is selected via ``sfa_id`` / ``sfb_id`` for
          ``scale_vec::1X`` / ``2X``):

          - **4X** (NVFP4 / MXFP4 with block_size=16, K=64): all 4 bytes are
            used, ``sfa_id = sfb_id = 0``.
          - **2X** (MXFP4 with block_size=32, K=64): 2 bytes per cell per inst,
            ``sfa_id ∈ {0, 2}`` (lower / upper half-word).
          - **1X** (MXFP8/MXFP6/MXFP4-mixed with block_size=32, K=32):
            1 byte per cell per inst, ``sfa_id ∈ {0, 1, 2, 3}`` (byte position).

          For 1X / 2X the unused byte slots can hold SFs for **other** inst-K
          iterations — when the kernel rotates ``sfa_id`` across multiple
          ``mma_scaled`` calls with the *same* ``sfa`` tensor (the standard
          time-multiplex trick), one shared TMEM cell-round serves up to 4
          inst-K iters.

        How to populate ``sfa`` / ``sfb``
        --------------------------------

        The intended pattern is:

        1. Pre-shuffle the SF tensor in HBM into the canonical block-scaled
           layout (use :func:`tilus.testing.shuffle_sf_to_block_scaled_layout`
           or have the quantizer emit it directly).
        2. ``self.tma.global_to_shared(...)`` to load it into SMEM (the canonical
           HBM layout maps 1:1 to the SMEM atom for ``tcgen05.cp.32x128b.warpx4``).
        3. ``self.tcgen05.copy(src=s_sf, dst=t_sf, multicast="warpx4")`` to copy
           into TMEM.
        4. ``self.tcgen05.mma_scaled(...)`` with the now-populated ``t_sf``.

        Parameters
        ----------
        a : SharedTensor | TMemoryTensor
            Per-CTA A operand of shape ``(M, K)``. Dtype must be in the row's
            ``a_dtypes`` set (see support matrix).
        b : SharedTensor
            Per-CTA B operand of shape ``(K, N)``. Dtype must be in the row's
            ``b_dtypes`` set.
        d : TMemoryTensor
            Per-CTA D accumulator. Shape ``(M, N)`` for ``cta_group=1``,
            ``(M, 2N)`` for ``cta_group=2``. Dtype is FP32 (or FP16 — encoded in
            inst-desc bits 4–5).
        sfa : TMemoryTensor
            Per-CTA scale tensor for A, shape ``[32, M/32, 4]``,
            ``WARPX4`` duplication. Dtype f8e4m3 (UE4M3) or f8e8m0 (UE8M0)
            depending on the row.
        sfb : TMemoryTensor
            Per-CTA scale tensor for B, shape ``[32, N/32, 4]``, same dtype as
            ``sfa``.
        sf_block_size : int
            Block size in K-elements covered by one SF: 16 or 32. With
            ``a.dtype`` and ``sfa.dtype``, this selects the row of the support
            matrix.
        enable_input_d : Expr | bool
            If true, ``D = A*B + D`` (accumulating). If false, ``D = A*B``.
        cta_group : int, default 1
            ``1`` (single-CTA inst) or ``2`` (2-CTA cluster inst — see
            partitioning table).
        sfa_id : int, default 0
            Which byte slot of the SFA TMEM cell this inst reads. Must be in
            ``valid_sf_ids`` for the matched row (e.g. ``{0}`` for 4X,
            ``{0, 2}`` for 2X, ``{0, 1, 2, 3}`` for 1X).
        sfb_id : int, default 0
            Same as ``sfa_id`` but for SFB.

        Notes
        -----
        - **Thread group**: must be executed by exactly 32 threads (a single
          warp). Use ``with self.single_warp():``.
        - **Hardware**: requires sm_100a (Blackwell datacenter).
        - **PTX**: lowers to *one* call of
          ``tcgen05.mma.cta_group::C.kind::K.block_scale.scale_vec::S``
          (PTX 9.7.16.10).
        """
        num_threads = self._builder.tg_stack.current_num_threads
        if num_threads != 32:
            raise InstructionError(
                "tcgen05.mma_scaled must be called by a warp (32 threads), got {} threads".format(num_threads)
            )
        if cta_group not in (1, 2):
            raise InstructionError("cta_group must be 1 or 2, got {}".format(cta_group))
        if sf_block_size not in (16, 32):
            raise InstructionError("sf_block_size must be 16 or 32, got {}".format(sf_block_size))
        if not isinstance(b, SharedTensor):
            raise InstructionError("mma_scaled requires b to be SharedTensor, got {}".format(type(b).__name__))
        if not isinstance(sfa, TMemoryTensor) or not isinstance(sfb, TMemoryTensor):
            raise InstructionError(
                "mma_scaled requires sfa and sfb to be TMemoryTensors, got {} and {}".format(
                    type(sfa).__name__, type(sfb).__name__
                )
            )
        if len(a.shape) != 2 or len(b.shape) != 2 or len(d.shape) != 2:
            raise InstructionError(
                "mma_scaled requires 2D a/b/d tensors; got shapes a={}, b={}, d={}".format(a.shape, b.shape, d.shape)
            )
        if len(sfa.shape) != 3 or len(sfb.shape) != 3:
            raise InstructionError(
                "mma_scaled requires 3D sfa/sfb tensors of shape [32, MN_fold, 4]; got sfa={}, sfb={}".format(
                    sfa.shape, sfb.shape
                )
            )

        # Look up the dispatch row from operand types and the user's choice.
        row = _lookup_block_scaled_row(
            a_dtype=a.dtype,
            b_dtype=b.dtype,
            sfa_dtype=sfa.dtype,
            sfb_dtype=sfb.dtype,
            sf_block_size=sf_block_size,
            cta_group=cta_group,
        )
        if row is None:
            raise InstructionError(
                "Unsupported block-scaled MMA combination: a.dtype={}, b.dtype={}, "
                "sfa.dtype={}, sfb.dtype={}, sf_block_size={}, cta_group={}. "
                "See the support matrix in Tcgen05InstructionGroup.mma_scaled docstring.".format(
                    a.dtype.name, b.dtype.name, sfa.dtype.name, sfb.dtype.name, sf_block_size, cta_group
                )
            )

        # Validate per-CTA M, N, K shapes against the matched row.
        M, K = a.shape[0], a.shape[1]
        N = b.shape[1]
        if M not in row.M_set:
            raise InstructionError(
                "mma_scaled per-CTA M={} not in {} for kind::{}, scale_vec::{}, cta_group={}.".format(
                    M, sorted(row.M_set), row.kind, row.scale_vec, cta_group
                )
            )
        if N < row.N_min or N > row.N_max or N % row.N_step != 0:
            raise InstructionError(
                "mma_scaled per-CTA N={} not in [{}, {}] step {} for kind::{}, scale_vec::{}, cta_group={}.".format(
                    N, row.N_min, row.N_max, row.N_step, row.kind, row.scale_vec, cta_group
                )
            )
        if K != row.inst_K:
            raise InstructionError(
                "mma_scaled per-CTA K={} != inst_K={} for kind::{}, scale_vec::{}.".format(
                    K, row.inst_K, row.kind, row.scale_vec
                )
            )
        if b.shape[0] != K:
            raise InstructionError(
                "mma_scaled requires a.shape[1] == b.shape[0]; got a={}, b={}".format(a.shape, b.shape)
            )

        # D shape: (M, N) for cta=1, (M, 2N) for cta=2.
        expected_d = (M, N) if cta_group == 1 else (M, 2 * N)
        if tuple(d.shape) != expected_d:
            raise InstructionError(
                "mma_scaled per-CTA d shape mismatch: expected {}, got {} (cta_group={})".format(
                    expected_d, tuple(d.shape), cta_group
                )
            )

        # SFA / SFB shapes.
        if tuple(sfa.shape) != (32, M // 32, 4):
            raise InstructionError(
                "mma_scaled sfa shape mismatch: expected [32, {}, 4] (= [32, M/32, 4]), got {}".format(
                    M // 32, list(sfa.shape)
                )
            )
        if tuple(sfb.shape) != (32, N // 32, 4):
            raise InstructionError(
                "mma_scaled sfb shape mismatch: expected [32, {}, 4] (= [32, N/32, 4]), got {}".format(
                    N // 32, list(sfb.shape)
                )
            )

        # SFA_ID / SFB_ID validity per scale_vec.
        if sfa_id not in row.valid_sf_ids:
            raise InstructionError(
                "sfa_id={} not in {} for scale_vec::{}".format(sfa_id, list(row.valid_sf_ids), row.scale_vec)
            )
        if sfb_id not in row.valid_sf_ids:
            raise InstructionError(
                "sfb_id={} not in {} for scale_vec::{}".format(sfb_id, list(row.valid_sf_ids), row.scale_vec)
            )

        # Dispatch to the SS or TS builder. The IR carries the fully-resolved
        # (kind, scale_vec) pair — emitter does no further dispatch.
        if isinstance(a, SharedTensor):
            self._builder.tcgen05_mma_scaled_ss(
                a=a, b=b, d=d, sfa=sfa, sfb=sfb,
                kind=row.kind, scale_vec=row.scale_vec,
                sf_block_size=sf_block_size,
                sfa_id=sfa_id, sfb_id=sfb_id,
                cta_group=cta_group,
                enable_input_d=enable_input_d,
            )
        elif isinstance(a, TMemoryTensor):
            self._builder.tcgen05_mma_scaled_ts(
                a=a, b=b, d=d, sfa=sfa, sfb=sfb,
                kind=row.kind, scale_vec=row.scale_vec,
                sf_block_size=sf_block_size,
                sfa_id=sfa_id, sfb_id=sfb_id,
                cta_group=cta_group,
                enable_input_d=enable_input_d,
            )
        else:
            raise InstructionError(
                "mma_scaled requires a to be SharedTensor or TMemoryTensor, got {}".format(type(a).__name__)
            )
