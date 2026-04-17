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
from __future__ import annotations

from dataclasses import dataclass

from tilus.ir.inst import Instruction


@dataclass(frozen=True, eq=False)
class FenceProxyAsync(Instruction):
    """Bidirectional async proxy fence: fence.proxy.async.{space}."""

    space: str

    @staticmethod
    def create(space: str) -> FenceProxyAsync:
        assert space in ("shared", "global"), (
            f"Invalid space for async proxy fence: {space}. Supported candidates are 'shared' and 'global'."
        )
        return FenceProxyAsync(output=None, inputs=(), space=space)


@dataclass(frozen=True, eq=False)
class FenceProxyAsyncRelease(Instruction):
    """Unidirectional generic-to-async release proxy fence.

    PTX: fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster

    This is a lighter-weight alternative to the bidirectional fence.proxy.async.shared::cta.
    It only ensures that prior generic proxy writes to shared::cta memory are visible to
    subsequent async proxy reads, with release semantics at cluster scope.
    """

    @staticmethod
    def create() -> FenceProxyAsyncRelease:
        return FenceProxyAsyncRelease(output=None, inputs=())
