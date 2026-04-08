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


from .root import InstructionGroup


class FenceInstructionGroup(InstructionGroup):
    """Fence instructions for memory ordering.

    Naming convention follows the PTX fence instruction taxonomy
    (PTX ISA 9.7.13.4: membar / fence). Each tilus method corresponds
    to one PTX instruction template with parameters for meaningful variation.

    Implemented
    -----------
    proxy_async(space)
        Bidirectional async proxy fence: ``fence.proxy.async.{space}``
    proxy_async_release()
        Unidirectional generic-to-async release:
        ``fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster``

    TODO
    ----
    thread(sem, scope, sync_restrict=None)
        Thread fence: ``fence.{sem}.{scope}``
        When sync_restrict is set:
        - ``fence.acquire.sync_restrict::shared::cluster.cluster``
        - ``fence.release.sync_restrict::shared::cta.cluster``
    proxy_alias()
        Bidirectional alias proxy fence: ``fence.proxy.alias``
    proxy_async_acquire()
        Unidirectional generic-to-async acquire:
        ``fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster``
    proxy_tensormap_release(scope)
        Unidirectional tensormap release:
        ``fence.proxy.tensormap::generic.release.{scope}``
    proxy_tensormap_acquire(addr, scope)
        Unidirectional tensormap acquire:
        ``fence.proxy.tensormap::generic.acquire.{scope} [addr], 128``
    """

    def proxy_async(self, space: str = "shared") -> None:
        """Bidirectional async proxy fence.

        PTX: ``fence.proxy.async.{space}``

        Establishes ordering between the async proxy and the generic proxy
        for memory operations in the specified state space.
        """
        if space not in ("shared", "global"):
            raise ValueError(
                f"Invalid scope for async proxy fence: {space}. Supported candidates are 'shared' and 'global'."
            )
        self._builder.fence_proxy_async(space=space)

    def proxy_async_release(self) -> None:
        """Unidirectional generic-to-async release proxy fence for shared memory.

        PTX: ``fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster``

        A lighter-weight alternative to proxy_async(). Only ensures that prior
        generic proxy writes to shared::cta memory are visible to subsequent
        async proxy reads (e.g., stmatrix followed by tma.shared_to_global).

        Requires sm_90 or higher.
        """
        self._builder.fence_proxy_async_release()
