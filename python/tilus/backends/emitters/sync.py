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
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import SyncReduceThreadsInst, SyncThreadsInst


@register_emitter(SyncThreadsInst)
class SyncThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncThreadsInst) -> None:
        self.sync()


@register_emitter(SyncReduceThreadsInst)
class SyncReduceThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncReduceThreadsInst) -> None:
        self.declare(inst.var, init=self.sync_reduce(inst.reduce_value, op=inst.reduce_op))
