"""
Query or lock GPU clocks.

Usage:
    python lock-gpu-clocks.py --gpu <gpu_index> [--reset | --clock <base|max>]


Examples
--------
# Query the current GPU 0 clocks:
python lock-gpu-clocks.py

# Lock the GPU 0 clocks to base frequency:
python lock-gpu-clocks.py --clock base

# Lock the GPU 0 clocks to maximum frequency:
python lock-gpu-clocks.py --clock max

# Reset the GPU 0 clocks to default settings:
python lock-gpu-clocks.py --reset

# Lock the GPU 1 clocks to maximum frequency:
python lock-gpu-clocks.py --gpu 1 --clock max

"""

import argparse
import os
import subprocess
import sys
import time

from cuda.bindings import runtime
from pynvml import (
    NVML_CLOCK_ID_CURRENT,
    NVML_CLOCK_LIMIT_ID_TDP,
    NVML_CLOCK_SM,
    NVML_FEATURE_DISABLED,
    NVML_FEATURE_ENABLED,
    nvmlDeviceGetClock,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceResetGpuLockedClocks,
    nvmlDeviceSetGpuLockedClocks,
    nvmlDeviceSetPersistenceMode,
    nvmlInit,
    nvmlShutdown,
)

parser = argparse.ArgumentParser(description="Lock GPU clocks to a specific frequency.")
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU index to lock the clocks for (default: 0)",
)
parser.add_argument(
    "--reset",
    "-r",
    action="store_true",
    help="Reset the GPU clocks to default settings",
)
parser.add_argument(
    "--clock",
    "-c",
    type=str,
    default=None,
    choices=["base", "max"],
    help="Lock the GPU clocks to base or maximum frequency",
)

args = parser.parse_args()


def query_gpu_clocks(handle):
    base_clock = nvmlDeviceGetClock(handle, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)
    print(f"GPU {args.gpu} current clocks: {base_clock} MHz")


def main():
    if not args.reset and args.clock is None:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(args.gpu)
        query_gpu_clocks(handle)
        nvmlShutdown()
        exit(0)

    if args.reset and args.clock is not None:
        print("Please specify either --reset or --clock <base|max>, not both.")
        sys.exit(0)

    if os.geteuid() != 0:
        # we require root privileges to set GPU clocks
        command = "sudo -E {python} {script} --gpu {gpu} {clock} {reset}".format(
            python=sys.executable,
            script=os.path.abspath(__file__),
            gpu=args.gpu,
            clock="--clock " + args.clock if args.clock else "",
            reset="--reset" if args.reset else "",
        )
        subprocess.run(command.split(), env=os.environ, shell=False)
        return

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.gpu)
    if args.reset:
        nvmlDeviceResetGpuLockedClocks(handle)
        nvmlDeviceSetPersistenceMode(handle, NVML_FEATURE_DISABLED)
    elif args.clock:
        # set persistent mode
        nvmlDeviceResetGpuLockedClocks(handle)
        nvmlDeviceSetPersistenceMode(handle, NVML_FEATURE_ENABLED)
        if args.clock == "base":
            nvmlDeviceSetGpuLockedClocks(handle, NVML_CLOCK_LIMIT_ID_TDP, NVML_CLOCK_LIMIT_ID_TDP)
        elif args.clock == "max":
            error, prop = runtime.cudaGetDeviceProperties(args.gpu)
            if error != 0:
                print(f"Error getting device properties: {error}")
                sys.exit(1)
            max_clock = prop.clockRate
            nvmlDeviceSetGpuLockedClocks(handle, max_clock, max_clock)
        else:
            assert False
    else:
        assert False

    time.sleep(0.10)  # wait until the clocks are changed
    query_gpu_clocks(handle)  # print the current GPU clocks
    nvmlShutdown()


if __name__ == "__main__":
    main()
