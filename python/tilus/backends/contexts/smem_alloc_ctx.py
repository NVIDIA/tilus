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

from tilus.backends.context import BaseEmitContext
from tilus.hidet.ir import StmtBuilder
from tilus.hidet.ir.dtypes import uint8
from tilus.hidet.ir.expr import Expr, Var
from tilus.hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from tilus.ir.tensor import SharedTensor
from tilus.target import get_current_target


class SharedMemoryAllocator:
    def __init__(self) -> None:
        self.free_slots: list[tuple[int, int]] = [(0, (1 << 32) - 1)]
        self.addr2nbytes: dict[int, int] = {}
        self.allocated: int = 0
        self.maximum_allocated: int = 0

    def allocate(self, nbytes: int, alignment: int = 128) -> int:
        # align the nbytes to the requested alignment
        nbytes = (nbytes + alignment - 1) // alignment * alignment

        # find the first slot that can fit the request (with alignment)
        for i, (start, end) in enumerate(self.free_slots):
            aligned_start = (start + alignment - 1) // alignment * alignment
            if aligned_start + nbytes <= end:
                break
        else:
            raise RuntimeError("No free shared memory slot available")
        addr = aligned_start
        # Split the slot: [start, aligned_start) is leftover before, [addr+nbytes, end) is leftover after
        del self.free_slots[i]
        if start < addr:
            self.free_slots.append((start, addr))
        if addr + nbytes < end:
            self.free_slots.append((addr + nbytes, end))
        self.free_slots.sort(key=lambda x: x[0])
        self.addr2nbytes[addr] = nbytes
        self.maximum_allocated = max(self.maximum_allocated, addr + nbytes)
        self.allocated += nbytes
        return addr

    def free(self, addr: int) -> None:
        # find the slot that is right before the address
        before = [i for i, slot in enumerate(self.free_slots) if slot[1] <= addr]
        after = [i for i, slot in enumerate(self.free_slots) if slot[0] > addr]
        assert len(before) + len(after) == len(self.free_slots)
        nbytes = self.addr2nbytes[addr]
        if (
            before
            and after
            and self.free_slots[before[-1]][1] == addr
            and self.free_slots[after[0]][0] == addr + nbytes
        ):
            # merge three slots
            self.free_slots[before[-1]] = (self.free_slots[before[-1]][0], self.free_slots[after[0]][1])
        elif before and self.free_slots[before[-1]][1] == addr:
            # merge with previous slot
            self.free_slots[before[-1]] = (self.free_slots[before[-1]][0], addr + nbytes)
        elif after and self.free_slots[after[0]][0] == addr + nbytes:
            # merge with next slot
            self.free_slots[after[0]] = (addr, self.free_slots[after[0]][1])
        else:
            # add a new slot
            self.free_slots.append((addr, addr + nbytes))
            self.free_slots = list(sorted(self.free_slots, key=lambda x: x[0]))
        self.allocated -= nbytes
        del self.addr2nbytes[addr]


class SharedMemoryAllocationContext(BaseEmitContext):
    def __post_init__(self):
        # shared memory allocator
        self.smem_allocator: SharedMemoryAllocator = SharedMemoryAllocator()

        # mapping from shared value to the address in shared memory allocator for all allocated shared values
        self.shared_value_allocator_addr: dict[SharedTensor, int] = {}

        # maximum shared workspace bytes requested by all instructions
        self.shared_workspace_var: Optional[Var] = None
        self.shared_workspace_bytes: int = 0

    def allocate_shared_tensor(self, tensor: SharedTensor, nbytes: int) -> int:
        alignment = self._get_swizzle_alignment(tensor)
        addr: int = self.smem_allocator.allocate(nbytes, alignment=alignment)
        assert tensor not in self.shared_value_allocator_addr
        self.shared_value_allocator_addr[tensor] = addr
        return addr

    @staticmethod
    def _get_swizzle_alignment(tensor: SharedTensor) -> int:
        """Compute the required shared memory alignment for a tensor based on its swizzle mode.

        When TMA copies data to/from shared memory with swizzle enabled, the hardware applies
        the swizzle pattern relative to the absolute shared memory address. If the tensor's base
        address is not aligned to the swizzle repeat boundary, there is an offset between the
        software swizzle (applied to local element offsets) and the hardware swizzle (applied to
        absolute addresses), causing data corruption.

        The swizzle repeat period (in bytes) depends on the swizzle mode:
          - CU_TENSOR_MAP_SWIZZLE_32B:  repeats every  2 * 128 =  256 bytes
          - CU_TENSOR_MAP_SWIZZLE_64B:  repeats every  4 * 128 =  512 bytes
          - CU_TENSOR_MAP_SWIZZLE_128B: repeats every  8 * 128 = 1024 bytes

        In our case, the Swizzle field in the SharedLayout encodes the element-based swizzling.

        See: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
             Section "Swizzle Pattern Pointer Offset Computation"
        """
        if not tensor.has_layout():
            return 128  # default alignment

        swizzle = tensor.layout.optional_swizzle
        if swizzle is None:
            return 128  # no swizzle, default 128-byte alignment

        repeat_bytes = (2 ** (swizzle.base + swizzle.bits + swizzle.shift)) * tensor.dtype.nbytes
        if repeat_bytes < 128:
            return 128  # minimum alignment is 128 bytes
        return repeat_bytes

    def free_shared_tensor(self, tensor: SharedTensor) -> None:
        assert tensor in self.shared_value_allocator_addr
        self.smem_allocator.free(addr=self.shared_value_allocator_addr[tensor])
        del self.shared_value_allocator_addr[tensor]

    def request_shared_workspace(self, nbytes: int) -> Expr:
        if self.shared_workspace_var is None:
            self.shared_workspace_var = Var("shared_workspace", type=~uint8)
            self.shared_workspace_bytes = nbytes
        else:
            self.shared_workspace_bytes = max(self.shared_workspace_bytes, nbytes)
        return self.shared_workspace_var

    def finalize(self):
        maximum_allocated = self.smem_allocator.maximum_allocated
        target = get_current_target()

        # define the shared workspace variable if needed
        if self.shared_workspace_var is not None:
            workspace_offset = (maximum_allocated + 127) // 128 * 128  # align to 128 bytes
            maximum_allocated += self.shared_workspace_bytes
            sb = StmtBuilder()
            sb.declare(self.shared_workspace_var, init=dynamic_shared_memory(workspace_offset, dtype=uint8))
            self.kernel_prepend(sb.finish())

        # set the dynamic shared memory size
        if target.is_nvgpu():
            self.codegen.builder.extend_attrs({"cuda.dynamic_smem_bytes": maximum_allocated})
        else:
            raise NotImplementedError()
