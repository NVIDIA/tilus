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
from typing import Any, Optional

from tilus.ir.builders import StmtBuilder


class TilusContext:
    def __enter__(self) -> Optional[Any]:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ThreadGroupContext(TilusContext):
    def __init__(self, builder: StmtBuilder, thread_begin: int, num_threads: int):
        self.builder: StmtBuilder = builder
        self.thread_begin: int = thread_begin
        self.num_threads: int = num_threads

        self.ctx = self.builder.thread_group(thread_begin=thread_begin, num_threads=num_threads)

    def __enter__(self) -> None:
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.__exit__(exc_type, exc_val, exc_tb)
