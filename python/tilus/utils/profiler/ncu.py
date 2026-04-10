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
from typing import Any

from tilus.utils.profiler.common import Profiler, ProfilerReport, profiler_main

_ncu_template = """
{profiler_path}
--export {report_path}
--kernel-name regex:"{kernel_regex}"
--force-overwrite
--set full
--rule CPIStall
--rule FPInstructions
--rule HighPipeUtilization
--rule IssueSlotUtilization
--rule LaunchConfiguration
--rule Occupancy
--rule PCSamplingData
--rule SOLBottleneck
--rule SOLFPRoofline
--rule SharedMemoryConflicts
--rule SlowPipeLimiter
--rule ThreadDivergence
--rule UncoalescedGlobalAccess
--rule UncoalescedSharedAccess
--import-source yes
--check-exit-code yes
{python_executable} {python_script} {args}
""".replace("\n", " ").strip()

_profiler = Profiler(
    binary_name="ncu",
    ui_binary_name="ncu-ui",
    command_template=_ncu_template,
    report_dir="ncu-reports",
    report_ext="ncu-rep",
    display_name="Nsight Compute",
    entry_script=__file__,
)

NsightComputeReport = ProfilerReport


def ncu_run(func: Any, *args: Any, kernel_regex: str = ".*", **kwargs: Any) -> ProfilerReport:
    return _profiler.run(func, args, kwargs, extra_template_kwargs={"kernel_regex": kernel_regex})


if __name__ == "__main__":
    profiler_main()
