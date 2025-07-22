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
import shutil
from pathlib import Path

import tilus.option


def clear_cache(*items: str) -> None:
    """
    Clear the cache directory for the given items.

    Parameters
    ----------
    items: sequence[str]
        The path items append to the cache directory to determine the directory to clear.
    """
    root = Path(tilus.option.get_option("cache_dir")).resolve()
    dir_to_clear = root / Path(*items)
    print("Clearing tilus cache dir: {}".format(dir_to_clear))
    dir_to_clear.mkdir(parents=True, exist_ok=True)
    for item in dir_to_clear.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
