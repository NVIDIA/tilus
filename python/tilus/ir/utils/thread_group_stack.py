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
        self.group_index: list[int] = []
        self.group_size: list[int] = []

        self.thread_begin: list[int] = []
        self.thread_end: list[int] = []

    def stack_depth(self):
        return len(self.group_index)

    def push(self, group_index: int, group_size: int) -> None:
        depth = self.stack_depth()
        if depth > 0:
            parent_group_size = self.group_size[-1]
            if parent_group_size % group_size != 0:
                raise ValueError("group_size must be a divisor of the parent group_size")
            num_groups = parent_group_size // group_size
            if group_index < 0 or group_index >= num_groups:
                raise ValueError(
                    "group_index must be in [0, num_groups), got group_index={}, num_groups={}".format(
                        group_index, num_groups
                    )
                )
        self.group_index.append(group_index)
        self.group_size.append(group_size)

        if depth > 0:
            parent_group_size = self.group_size[-1]
            parent_thread_begin = self.thread_begin[-1]
            self.thread_begin.append(parent_thread_begin + group_index * group_size)
            self.thread_end.append(parent_thread_begin + (group_index + 1) * group_size)
        else:
            self.thread_begin.append(0)
            self.thread_end.append(group_size)

    def pop(self):
        self.group_index.pop()
        self.group_size.pop()
        self.thread_begin.pop()
        self.thread_end.pop()
