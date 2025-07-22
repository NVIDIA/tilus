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
from typing import Sequence

from hidet.ir.module import IRModule


def merge_ir_modules(modules: Sequence[IRModule]) -> IRModule:
    if len(modules) == 0:
        return IRModule()
    merged = modules[0].copy()
    for module in modules[1:]:
        if module.namespace != merged.namespace:
            raise ValueError("Cannot merge IRModules with different namespaces")
        # merge global vars
        for name, var in module.global_vars.items():
            if name in merged.global_vars:
                raise ValueError("Global variable {} has already existed in module.".format(name))
            merged.global_vars[name] = var
        # merge functions
        for name, func in module.functions.items():
            if name in merged.functions:
                raise ValueError("Function {} has already existed in module.".format(name))
            merged.functions[name] = func
        # merge extern functions
        for name, var in module.extern_functions.items():
            if name in merged.extern_functions:
                continue
            merged.extern_functions[name] = var

        # merge include headers, include_dirs, linking_dirs, linking_libs, object_files
        merged.include_headers.extend(
            [header for header in module.include_headers if header not in merged.include_dirs]
        )
        merged.include_dirs.extend([dir for dir in module.include_dirs if dir not in merged.include_dirs])
        merged.linking_dirs.extend([dir for dir in module.linking_dirs if dir not in merged.linking_dirs])
        merged.linking_libs.extend([lib for lib in module.linking_libs if lib not in merged.linking_libs])
        merged.object_files.extend([file for file in module.object_files if file not in merged.object_files])

    return merged
