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
class ThreadGroupStack:
    def __init__(self):
        self.num_threads: list[int] = []
        self.thread_begin: list[int] = []
        self.thread_end: list[int] = []
    
    @property
    def current_num_threads(self) -> int:
        if self.stack_depth() == 0:
            raise ValueError("Thread group stack is empty")
        return self.num_threads[-1]

    @property
    def current_thread_begin(self) -> int:
        if self.stack_depth() == 0:
            raise ValueError("Thread group stack is empty")
        return self.thread_begin[-1]

    @property
    def current_thread_end(self) -> int:
        if self.stack_depth() == 0:
            raise ValueError("Thread group stack is empty")
        return self.thread_end[-1]

    def stack_depth(self):
        return len(self.num_threads)

    def push(self, thread_begin: int, num_threads: int) -> None:
        depth = self.stack_depth()
        if depth > 0:
            parent_num_threads = self.num_threads[-1]
            if parent_num_threads % num_threads != 0:
                raise ValueError("group_size must be a divisor of the parent group_size")
            if thread_begin < 0 or thread_begin + num_threads > parent_num_threads:
                raise ValueError(
                    "thread_begin must be in [0, parent_num_threads - num_threads), got thread_begin={}, num_threads={}, parent_num_threads={}".format(
                        thread_begin, num_threads, parent_num_threads
                    )
                )
        self.num_threads.append(num_threads)

        if depth > 0:
            parent_num_threads = self.num_threads[-1]
            parent_thread_begin = self.thread_begin[-1]
            self.thread_begin.append(parent_thread_begin + thread_begin)
            self.thread_end.append(parent_thread_begin + thread_begin + num_threads)
        else:
            self.thread_begin.append(0)
            self.thread_end.append(num_threads)

    def pop(self):
        self.num_threads.pop()
        self.thread_begin.pop()
        self.thread_end.pop()
