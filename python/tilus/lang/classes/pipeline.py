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
import tilus
from tilus import int32, uint32
from tilus.ir.tensor import RegisterTensor


class Pipeline(tilus.Class):
    def __init__(self, num_stages: int, producer_arrive_count: int, consumer_arrive_count: int, debug: bool = False):
        self.num_stages: int = num_stages
        self._full_barriers = self.mbarrier.alloc([consumer_arrive_count for _ in range(num_stages)])
        self._empty_barriers = self.mbarrier.alloc([producer_arrive_count for _ in range(num_stages)])
        self.producer_stage: int32 = 0
        self.consumer_stage: int32 = 0
        self._producer_phase: uint32 = self.mbarrier.producer_initial_phase
        self._consumer_phase: uint32 = self.mbarrier.consumer_initial_phase

        # used for debugging
        self._name: str = self.__class__.__name__
        self._debug: bool = debug

    def producer_acquire(self):
        if self._debug:
            self.printf(
                "%20s: Producer[%3d, %3d) acquiring stage %d\n",
                self._name,
                self.current_thread_begin,
                self.current_thread_end,
                self.producer_stage,
            )
        self.mbarrier.wait(barrier=self._full_barriers[self.producer_stage], phase=self._producer_phase)
        if self._debug:
            self.printf(
                "%20s: Producer[%3d, %3d) acquired stage %d\n",
                self._name,
                self.current_thread_begin,
                self.current_thread_end,
                self.producer_stage,
            )

    def producer_advance(self):
        if self._debug:
            self.printf(
                "%20s: Producer[%3d, %3d) advancing stage %d\n",
                self._name,
                self.current_thread_begin,
                self.current_thread_end,
                self.producer_stage,
            )
        self.producer_stage = (self.producer_stage + 1) % self.num_stages
        self._producer_phase = self._producer_phase ^ (self.producer_stage == 0)

    def producer_release_barrier(self) -> RegisterTensor:
        return self._empty_barriers[self.producer_stage]

    def consumer_acquire(self):
        if self._debug:
            self.printf(
                "%20s: Consumer[%3d, %3d) acquiring stage %d\n",
                self._name,
                self.current_thread_begin,
                self.current_thread_end,
                self.consumer_stage,
            )
        self.mbarrier.wait(barrier=self._empty_barriers[self.consumer_stage], phase=self._consumer_phase)
        if self._debug:
            self.printf(
                "%20s: Consumer[%3d, %3d) acquired stage %d\n",
                self._name,
                self.current_thread_begin,
                self.current_thread_end,
                self.consumer_stage,
            )

    def consumer_advance(self):
        if self._debug:
            self.printf(
                "%20s: Consumer[%3d, %3d) advancing stage %d\n",
                self._name,
                self.current_thread_begin,
                self.current_thread_end,
                self.consumer_stage,
            )
        self.consumer_stage = (self.consumer_stage + 1) % self.num_stages
        self._consumer_phase = self._consumer_phase ^ (self.consumer_stage == 0)

    def consumer_release_barrier(self) -> RegisterTensor:
        return self._full_barriers[self.consumer_stage]
