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
    """Fence instructions for memory ordering between memory proxies.

    CUDA GPUs have multiple memory proxies (generic, async, alias, tensormap) that can access
    the same memory through different paths. When one proxy writes data that another proxy needs
    to read, a **proxy fence** is required to ensure the writes are visible.

    The most common scenario is coordinating between:

    - **Generic proxy** writes: ``store_shared()``, register-to-shared stores
    - **Async proxy** reads/writes: TMA operations (``tma.global_to_shared()``, ``tma.shared_to_global()``)

    For example, after writing to shared memory with ``store_shared()`` and before issuing
    ``tma.shared_to_global()``, a ``proxy_async()`` or ``proxy_async_release()`` fence is needed
    to ensure the TMA engine sees the updated data.

    ``proxy_async_release()`` is a lighter-weight alternative to ``proxy_async()`` when only
    generic-to-async ordering is needed (not bidirectional).
    """

    def proxy_async(self, space: str = "shared") -> None:
        """Bidirectional async proxy fence.

        Establishes ordering between the async proxy and the generic proxy for memory
        operations in the specified state space. Required when both proxies access the
        same memory (e.g., generic writes followed by TMA reads, or TMA writes followed
        by generic reads).

        Parameters
        ----------
        space: str
            The state space for the fence. Candidates: ``'shared'``, ``'global'``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``fence.proxy.async.shared::cta`` or ``fence.proxy.async.global``
        """
        if space not in ("shared", "global"):
            raise ValueError(
                f"Invalid scope for async proxy fence: {space}. Supported candidates are 'shared' and 'global'."
            )
        self._builder.fence_proxy_async(space=space)

    def proxy_async_release(self) -> None:
        """Unidirectional generic-to-async release proxy fence for shared memory.

        A lighter-weight alternative to ``proxy_async()``. Only ensures that prior generic proxy
        writes to shared memory are visible to subsequent async proxy reads (e.g.,
        ``store_shared()`` followed by ``tma.shared_to_global()``).

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 9.0+ (sm_90).
        - **PTX**: ``fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster``
        """
        self._builder.fence_proxy_async_release()
