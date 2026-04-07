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
from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.dtypes import boolean, int32
from tilus.hidet.ir.expr import equal
from tilus.hidet.ir.primitives.cuda.ldst import load, store
from tilus.ir.instructions import LockSemaphoreInst, ReleaseSemaphoreInst


@register_emitter(LockSemaphoreInst)
class LockSemaphoreEmitter(BaseInstEmitter):
    def emit(self, inst: LockSemaphoreInst) -> None:
        semaphore = self.declare_var("semaphore", tp=~int32, init=inst.semaphore)
        semaphore_expect = self.declare_var("semaphore_expect", tp=int32, init=inst.value)

        if self.current_thread_group_depth == 1:
            # at the outermost thread group, we use sync_reduce to check if any thread has acquired the semaphore successfully
            # it's easy to generate warp-uniform code
            with self.while_loop(boolean.true):
                semaphore_value = self.declare_var("semaphore_value", tp=int32, init=-int32.one)
                with self.single_thread():
                    self.assign(semaphore_value, load(addr=semaphore, space="generic", sync="acquire", scope="gpu"))
                    cond = self.sync_reduce(equal(semaphore_value, semaphore_expect), op="or")  # type: ignore
                    with self.if_then(cond):
                        self.brk()
        else:
            # at some inner thread group, we cannot use sync_reduce since there is not such underlying instruction like syncrhoize_or for a
            # subset of threads.  Instead, we let a single thread to repeat loading the semaphore until it acquires the lock, then
            # synchronize with other threads to make sure all threads see the updated value of the semaphore.
            with self.single_thread():
                with self.while_loop(boolean.true):
                    semaphore_value = self.declare_var("semaphore_value", tp=int32, init=-int32.one)
                    self.assign(semaphore_value, load(addr=semaphore, space="generic", sync="acquire", scope="gpu"))
                    with self.if_then(equal(semaphore_value, semaphore_expect)):
                        self.brk()
            self.sync()


@register_emitter(ReleaseSemaphoreInst)
class ReleaseSemaphoreEmitter(BaseInstEmitter):
    def emit(self, inst: ReleaseSemaphoreInst) -> None:
        semaphore = self.declare_var("semaphore", tp=~int32, init=inst.semaphore)

        with self.single_thread():
            self.append(store(addr=semaphore, space="generic", value=inst.value, sync="release", scope="gpu"))
