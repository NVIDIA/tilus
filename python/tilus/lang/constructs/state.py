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
from tilus.lang.instructions import InstructionInterface


class Class(InstructionInterface):
    """A helper class for organizing kernel logic into reusable components.

    ``tilus.Class`` works like :class:`tilus.Script` but for helper objects that
    are not kernels themselves. It can allocate mbarriers, shared tensors, tensor
    memory, and use all tilus instructions.

    Subclass ``tilus.Class`` and define an ``__init__`` method to create reusable
    abstractions. The ``__init__`` is transpiled alongside the kernel that uses it.

    Example::

        class Pipeline(tilus.Class):
            def __init__(self, num_stages: int):
                self.barriers = self.mbarrier.alloc(counts=[1] * num_stages)
                self.stage: int32 = 0

            def advance(self):
                self.stage = (self.stage + 1) % self.num_stages

    Use it inside a :class:`tilus.Script`::

        class MyKernel(tilus.Script):
            def __call__(self, ...):
                pipe = Pipeline(num_stages=4)
                pipe.advance()
    """

    pass
