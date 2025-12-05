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

from hidet.ir.expr import Expr

from tilus.ir.inst import InstructionError
from tilus.ir.tensor import RegisterTensor, SharedTensor

from .root import InstructionGroup


class WgmmaInstructionGroup(InstructionGroup):
    def fence(self) -> None:
        self._builder.wgmma_fence()

    def commit_group(self) -> None:
        self._builder.wgmma_commit_group()

    def wait_group(self, n: Union[Expr, int]) -> None:
        self._builder.wgmma_wait_group(n)

    def mma(self, a: SharedTensor | RegisterTensor, b: SharedTensor, d: RegisterTensor) -> None:
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
