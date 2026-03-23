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
# Stripped for tilus integration — only library_paths discovery is needed.
import os
import os.path
from typing import Dict, Optional

library_paths: Dict[str, Optional[str]] = {"hidet": None, "hidet_runtime": None}


def _find_library_paths():
    """Find hidet runtime library from the original hidet installation or common paths."""
    if library_paths.get("hidet_runtime") is not None:
        return

    # Try to find from the original hidet installation
    try:
        import hidet.ffi.ffi as hidet_ffi

        hidet_ffi.load_library()
        library_paths.update(hidet_ffi.library_paths)
        return
    except (ImportError, OSError):
        pass

    # Search common locations
    from tilus.hidet.libinfo import get_library_search_dirs

    for library_dir in get_library_search_dirs():
        runtime_path = os.path.join(library_dir, "libhidet_runtime.so")
        if os.path.exists(runtime_path):
            library_paths["hidet_runtime"] = runtime_path
            hidet_path = os.path.join(library_dir, "libhidet.so")
            if os.path.exists(hidet_path):
                library_paths["hidet"] = hidet_path
            return


# Lazy initialization
_find_library_paths()
