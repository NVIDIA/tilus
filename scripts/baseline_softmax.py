# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pre-refactor baseline driver for the softmax kernel.

Runs the softmax example with a pinned autotune schedule so a single program
is compiled, dumping the Tilus and Hidet IR stages plus the generated
``source.cu`` into ``./cache``. After the refactor, re-run this script with
a fresh ``./cache`` directory and compare the artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples" / "softmax"))

import softmax  # noqa: E402
import tilus  # noqa: E402

tilus.option.cache_dir("./cache")
tilus.option.debug.dump_ir(True)

# Pin a single schedule so only one program is compiled.
softmax.FusedSoftmax.debug_schedule = dict(block_m=4, block_n=512, warps=8)

m, n = 4096, 4096
x = torch.randn(m, n, dtype=torch.float16, device="cuda")
y = torch.empty(m, n, dtype=torch.float16, device="cuda")

kernel = softmax.FusedSoftmax()
kernel(m, n, x, y)

torch.testing.assert_close(y, torch.softmax(x.float(), dim=1).to(torch.float16), atol=1e-2, rtol=1e-2)
print("OK softmax baseline captured at", (REPO / "cache").resolve())
