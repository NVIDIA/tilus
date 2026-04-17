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
from typing import Any

from tilus.utils.profiler.common import Profiler, ProfilerReport, profiler_main

_profiler = Profiler(
    binary_name="nsys",
    ui_binary_name="nsys-ui",
    command_template="{profiler_path} profile -o {report_path} {python_executable} {python_script} {args}",
    report_dir="nsys-reports",
    report_ext="nsys-rep",
    display_name="Nsight Systems",
    entry_script=__file__,
)

NsightSystemReport = ProfilerReport


def nsys_run(func: Any, *args: Any, **kwargs: Any) -> ProfilerReport:
    return _profiler.run(func, args, kwargs)


if __name__ == "__main__":
    profiler_main()
