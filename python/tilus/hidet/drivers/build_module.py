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
# Licensed under the Apache License, Version 2.0 (the "License")
# Stripped for tilus integration — only write_function_types is needed.
import os
import pickle
from typing import Dict

from tilus.hidet.ir.type import FuncType


def write_function_types(ir_module, output_dir):
    """Write function types for public functions in the IR module."""
    func_types: Dict[str, FuncType] = {
        func.name: FuncType.from_func(func) for func in ir_module.functions.values() if func.kind == "public"
    }
    with open(os.path.join(output_dir, "func_types.pickle"), "wb") as f:
        pickle.dump(func_types, f)
