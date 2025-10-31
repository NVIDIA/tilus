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
from hidet.utils.py import gcd, initialize, is_power_of_two, lcm, prod, same_list

from . import stats
from .bench_utils import benchmark_func
from .cache_utils import clear_cache
from .multiprocess import parallel_imap, parallel_map
from .py import cdiv, floor_log2, idiv, nbytes_from_nbits, relative_to_with_walk_up, to_snake_case
from .torch_utils import dtype_from_torch, dtype_to_torch
