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

from hidet.ir.expr import Expr

from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor

from .root import InstructionGroup


class TmaInstructionGroup(InstructionGroup):
    def global_to_shared(
        self,
        *,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        mbarrier: Expr | RegisterTensor,
        multicast_mask: Optional[Expr | int] = None,
        cache_policy: Optional[Expr] = None,
    ) -> None:
        """
        TMA async copy from global to shared tensor asynchronously.

        This instruction issues a TMA tensor async copy from global to shared tensor.

        The `src` parameter specifies the global tensor to copy from, while the `dst` parameter specifies the shared
        tensor to copy to.

        The `offsets` parameter specifies the starting offsets for each dimension of the global tensor where the tile
        will be copied from. The length of this sequence must match the number of dimensions of the global tensor.

        The `dims` parameter specifies which dimensions of the global tensor are being sliced. The dimension dim[0] of
        the global tensor corresponds to the first dimension of the shared tensor, dim[1] to the second, and so on.

        The `mbarrier` parameter specifies the memory barrier to be used for synchronizing the copy operation. It should be an uint32
        expression or a register tensor with one element of uint32 type specifying the address of the barrier in shared space.
        This instruction will signal an arrival to the barrier and increase the expect-tx by the number of bytes pending in the copy operation.
        Upon finishing the underlying TMA operation, the TMA engine will signal to reduce the expect-rx by the same number of bytes.

        The `multicast_mask` parameter specifies the multicast mask to be used. When not given, this instruction only
        copy the data from global memory to the shared memory of the current thread block. If given, it should be a
        uint16 variable specifying the multicast mask to be used.  When the i-th bit of the mask is set, the data will
        also be copied to the shared memory of the i-th thread block in the current thread block cluster.

        The `cache_policy` parameter specifies the cache policy to be used. It should be an uint64 variable encoded with the cache policy.

        Parameters
        ----------
        src: GlobalTensor
            The global tensor to copy from.
        dst: SharedTensor
            The shared tensor to copy to.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor where the tile will be copied from. The
            length of this sequence must match the number of dimensions of the global tensor.
        dims: Sequence[int]
            The dimensions of the global tensor that are being sliced. The length of this sequence must match the
            number of dimensions of the shared tensor being copied to.
        mbarrier: Expr | RegisterTensor
            The memory barrier to be used for synchronizing the copy operation. It should be an uint32 expression specifying the address
            of the barrier in shared space. It can also be a register tensor with a single element of uint32 type containing the address of the barrier.
        multicast_mask: Optional[Expr]
            The multicast mask to be used. It should be a uint16 variable specifying the multicast thread blocks. When not given, no multicast is performed.
        cache_policy: Optional[Expr]
            The cache policy to be used. It should be an uint64 variable encoded with the cache policy.
        """
        self._builder.copy_async_tensor_global_to_shared(
            src=src,
            dst=dst,
            offsets=offsets,
            dims=dims,
            mbarrier=mbarrier,
            multicast_mask=multicast_mask,
            cache_policy=cache_policy,
        )

    def shared_to_global(
        self,
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        cache_policy: Optional[Expr] = None,
    ) -> None:
        """
        TMA async copy from shared to global tensor asynchronously.

        This instruction issues a TMA tensor async copy from shared to global tensor.

        The `src` parameter specifies the shared tensor to copy from, while the `dst` parameter specifies the global
        tensor to copy to.

        The `offsets` parameter specifies the starting offsets for each dimension of the global tensor where the tile
        will be copied to. The length of this sequence must match the number of dimensions of the global tensor.

        The `dims` parameter specifies which dimensions of the global tensor are being sliced. The dimension dim[0] of
        the global tensor corresponds to the first dimension of the shared tensor, dim[1] to the second, and so on.

        The `cache_policy` parameter specifies the cache policy to be used. It should be an uint64 variable encoded with the cache policy.

        Parameters
        ----------
        src: SharedTensor
            The shared tensor to copy from.
        dst: GlobalTensor
            The global tensor to copy to.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor where the tile will be copied to. The
            length of this sequence must match the number of dimensions of the global tensor.
        dims: Sequence[int]
            The dimensions of the global tensor that are being sliced. The length of this sequence must match the
            number of dimensions of the shared tensor being copied from.
        cache_policy: Optional[Expr]
            The cache policy to be used. It should be an uint64 variable encoded with the cache policy.
        """
        self._builder.copy_async_tensor_shared_to_global(
            src=src, dst=dst, offsets=offsets, dims=dims, cache_policy=cache_policy
        )

    def commit_group(self):
        """
        Commit the previously issued async tensor copy operations.

        This instruction commits the previously issued async tensor copy operations.

        """
        self._builder.copy_async_tensor_commit_group()

    def wait_group(self, n: int) -> None:
        """
        Wait for the previously issued async tensor copy operations to complete.

        This instruction waits for the previously issued async tensor copy operations to complete.
        The `n` parameter specifies the number of groups to allow to be on-the-fly.

        Parameters
        ----------
        n: int
            The number of groups to allow to be on-the-fly. It should be an integer larger or equal to 0.
        """
        self._builder.copy_async_tensor_wait_group(n)

    def fence_proxy_copy_async(self):
        """
        Makes the modifications to shared tensors visible to TMA engine.

        This instruction makes the modifications to shared tensors visible to TMA engine. It should be in the thread group
        that has made modifications to shared tensors, and before copy the shared tensors to global memory with TMA-related instructions.
        """
        self._builder.fence_proxy_copy_async()
