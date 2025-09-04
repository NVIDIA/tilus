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
import functools
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class TargetProperties:
    compute_capability: Tuple[int, int] = (0, 0)
    feature_suffix: Optional[str] = None
    shared_memory_per_block: int = 0


@dataclass(frozen=True, eq=True)
class Target:
    kind: str
    arch: str
    properties: TargetProperties

    def __str__(self):
        return "{}/{}".format(self.kind, self.arch)

    def is_nvgpu(self):
        return self.kind == "nvgpu"

    def is_amdgpu(self):
        return self.kind == "amdgpu"

    def supports(self, target):
        """Check whether the current target supports the given target.

        This function checks whether all GPUs that are compatible with `self` are also compatible with `target`.

        Parameters
        ----------
        target:
            The target to be checked.

        Returns
        -------
        ret: bool
            Whether the GPUs that supports the current target also support the given target.
        """
        assert isinstance(target, Target)
        # whether the features supported by the self target contains features in the target
        if target == gpgpu_any:
            return True
        if self.kind != target.kind:
            return False

        self_major, self_minor = self.properties.compute_capability
        self_suffix = self.properties.feature_suffix
        target_major, target_minor = target.properties.compute_capability
        target_suffix = target.properties.feature_suffix

        if self.kind == "nvgpu":
            if target_suffix is None:
                return (self_major, self_minor) >= (target_major, target_minor)
            elif target_suffix == "f":
                return self_major == target_major and self_minor >= target_minor and (self_suffix in ("a", "f"))
            elif target_suffix == "a":
                return self_major == target_major and self_minor == target_minor and (self_suffix == "a")
            else:
                raise NotImplementedError(f"Unsupported target suffix: {target_suffix}")
        else:
            assert target_suffix is None and self_suffix is None
            return (self_major, self_minor) >= (target_major, target_minor)


"""
  Predefined targets
  
  The generic ones:
    - gpgpu/any: any GPU
    - amdgpu/any: any AMD GPU
    - nvgpu/any: any NVIDIA GPU
  are used to represent the generic targets that our compilation process (like scheduler) can work on.
  
  Each specific GPU must be represented by a specific target, e.g., amdgpu/gfx1100 for AMD RX 7900 XTX.
"""
gpgpu_any = Target(kind="gpgpu", arch="any", properties=TargetProperties())

"""
    AMD GPUs
"""
amdgpu_any = Target(
    kind="amdgpu",
    arch="any",
    properties=TargetProperties(compute_capability=(0, 0), feature_suffix=None, shared_memory_per_block=64 * 1024),
)

# RX 7900 XTX, etc.
amdgpu_gfx1100 = Target(
    kind="amdgpu",
    arch="gfx1100",
    properties=TargetProperties(compute_capability=(11, 0), feature_suffix=None, shared_memory_per_block=64 * 1024),
)

"""
    NVIDIA GPUs
    
    See Also: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities
    
    Suffixes:
    - No suffix: Base architecture
    - 'f': Family variant
    - 'a': Architecture-specific variant
"""
nvgpu_any = Target(
    "nvgpu", "any", TargetProperties(compute_capability=(0, 0), feature_suffix=None, shared_memory_per_block=64 * 1024)
)

# SM 7.0
nvgpu_sm70 = Target(
    "nvgpu", "sm70", TargetProperties(compute_capability=(7, 0), feature_suffix=None, shared_memory_per_block=96 * 1024)
)

# SM 7.5
nvgpu_sm75 = Target(
    "nvgpu", "sm75", TargetProperties(compute_capability=(7, 5), feature_suffix=None, shared_memory_per_block=96 * 1024)
)

# SM 8.0
nvgpu_sm80 = Target(
    "nvgpu",
    "sm80",
    TargetProperties(compute_capability=(8, 0), feature_suffix=None, shared_memory_per_block=163 * 1024),
)

# SM 8.6
nvgpu_sm86 = Target(
    "nvgpu", "sm86", TargetProperties(compute_capability=(8, 6), feature_suffix=None, shared_memory_per_block=99 * 1024)
)

# SM 8.7
nvgpu_sm87 = Target(
    "nvgpu", "sm87", TargetProperties(compute_capability=(8, 7), feature_suffix=None, shared_memory_per_block=99 * 1024)
)

# SM 8.8
nvgpu_sm88 = Target(
    "nvgpu", "sm88", TargetProperties(compute_capability=(8, 8), feature_suffix=None, shared_memory_per_block=99 * 1024)
)

# SM 8.9
nvgpu_sm89 = Target(
    "nvgpu", "sm89", TargetProperties(compute_capability=(8, 9), feature_suffix=None, shared_memory_per_block=99 * 1024)
)

# SM 9.0
nvgpu_sm90 = Target(
    "nvgpu",
    "sm90",
    TargetProperties(compute_capability=(9, 0), feature_suffix=None, shared_memory_per_block=227 * 1024),
)
nvgpu_sm90a = Target(
    "nvgpu",
    "sm90a",
    TargetProperties(compute_capability=(9, 0), feature_suffix="a", shared_memory_per_block=227 * 1024),
)

# SM 10.0
nvgpu_sm100 = Target(
    "nvgpu",
    "sm100",
    TargetProperties(compute_capability=(10, 0), feature_suffix=None, shared_memory_per_block=227 * 1024),
)
nvgpu_sm100f = Target(
    "nvgpu",
    "sm100f",
    TargetProperties(compute_capability=(10, 0), feature_suffix="f", shared_memory_per_block=227 * 1024),
)
nvgpu_sm100a = Target(
    "nvgpu",
    "sm100a",
    TargetProperties(compute_capability=(10, 0), feature_suffix="a", shared_memory_per_block=227 * 1024),
)

# SM 10.3
nvgpu_sm103 = Target(
    "nvgpu",
    "sm103",
    TargetProperties(compute_capability=(10, 3), feature_suffix=None, shared_memory_per_block=227 * 1024),
)
nvgpu_sm103f = Target(
    "nvgpu",
    "sm103f",
    TargetProperties(compute_capability=(10, 3), feature_suffix="f", shared_memory_per_block=227 * 1024),
)
nvgpu_sm103a = Target(
    "nvgpu",
    "sm103a",
    TargetProperties(compute_capability=(10, 3), feature_suffix="a", shared_memory_per_block=227 * 1024),
)

# SM 11.0
nvgpu_sm110 = Target(
    "nvgpu",
    "sm110",
    TargetProperties(compute_capability=(11, 0), feature_suffix=None, shared_memory_per_block=227 * 1024),
)
nvgpu_sm110f = Target(
    "nvgpu",
    "sm110f",
    TargetProperties(compute_capability=(11, 0), feature_suffix="f", shared_memory_per_block=227 * 1024),
)
nvgpu_sm110a = Target(
    "nvgpu",
    "sm110a",
    TargetProperties(compute_capability=(11, 0), feature_suffix="a", shared_memory_per_block=227 * 1024),
)

# SM 12.0
nvgpu_sm120 = Target(
    "nvgpu",
    "sm120",
    TargetProperties(compute_capability=(12, 0), feature_suffix=None, shared_memory_per_block=99 * 1024),
)
nvgpu_sm120f = Target(
    "nvgpu",
    "sm120f",
    TargetProperties(compute_capability=(12, 0), feature_suffix="f", shared_memory_per_block=99 * 1024),
)
nvgpu_sm120a = Target(
    "nvgpu",
    "sm120a",
    TargetProperties(compute_capability=(12, 0), feature_suffix="a", shared_memory_per_block=99 * 1024),
)

# SM 12.1
nvgpu_sm121 = Target(
    "nvgpu",
    "sm121",
    TargetProperties(compute_capability=(12, 1), feature_suffix=None, shared_memory_per_block=99 * 1024),
)
nvgpu_sm121f = Target(
    "nvgpu",
    "sm121f",
    TargetProperties(compute_capability=(12, 1), feature_suffix="f", shared_memory_per_block=99 * 1024),
)
nvgpu_sm121a = Target(
    "nvgpu",
    "sm121a",
    TargetProperties(compute_capability=(12, 1), feature_suffix="a", shared_memory_per_block=99 * 1024),
)


@functools.cache
def get_current_target() -> Target:
    from hidet import cuda

    has_nvgpu = cuda.available() and cuda.device_count() > 0
    # has_amdgpu = hip.available() and hip.device_count() > 0
    has_amdgpu = False

    if has_nvgpu and has_amdgpu:
        raise RuntimeError("Both AMD and NVIDIA GPUs are available. We do not support this configuration yet.")
    elif has_nvgpu:
        compute_capabilities = [cuda.compute_capability(i) for i in range(cuda.device_count())]
        if len(set(compute_capabilities)) > 1:
            raise RuntimeError(
                "Multiple NVIDIA GPUs with different compute capabilities are available. "
                "We do not support this configuration yet."
            )
        major, minor = cuda.compute_capability()

        nvgpu_targets = [
            nvgpu_sm70,
            nvgpu_sm75,
            nvgpu_sm80,
            nvgpu_sm86,
            nvgpu_sm87,
            nvgpu_sm88,
            nvgpu_sm89,
            nvgpu_sm90,
            nvgpu_sm90a,
            nvgpu_sm100,
            nvgpu_sm100f,
            nvgpu_sm100a,
            nvgpu_sm103,
            nvgpu_sm103f,
            nvgpu_sm103a,
            nvgpu_sm110,
            nvgpu_sm110f,
            nvgpu_sm110a,
            nvgpu_sm120,
            nvgpu_sm120f,
            nvgpu_sm120a,
            nvgpu_sm121,
            nvgpu_sm121f,
            nvgpu_sm121a,
        ]

        # Create target map based on (major, minor, suffix)
        target_map: dict[tuple[int, int, Optional[str]], Target] = {
            (t.properties.compute_capability[0], t.properties.compute_capability[1], t.properties.feature_suffix): t
            for t in nvgpu_targets
        }

        # first try arch-specific variant
        if (major, minor, "a") in target_map:
            return target_map[(major, minor, "a")]

        # then try family variant
        if (major, minor, "f") in target_map:
            return target_map[(major, minor, "f")]

        # finally try base architecture
        if (major, minor, None) in target_map:
            return target_map[(major, minor, None)]

        # If no target found, raise an error
        raise RuntimeError(f"Unsupported NVIDIA GPU compute capability: {major}.{minor}")
    elif has_amdgpu:
        from hidet import hip  # type: ignore

        compute_capabilities = [
            hip.compute_capability(i)
            for i in range(hip.device_count())
            if hip.properties(i).name.decode() != "AMD Radeon Graphics"  # skip the integrated GPU in AMD CPU
        ]
        if len(set(compute_capabilities)) > 1:
            raise RuntimeError(
                "Multiple AMD GPUs with different compute capabilities are available. "
                "We do not support this configuration yet."
            )
        major, minor = hip.compute_capability()
        amdgpu_targets = [amdgpu_gfx1100]
        target_map: dict[tuple[int, int], Target] = {  # type: ignore[no-redef]
            (t.properties.compute_capability[0], t.properties.compute_capability[1]): t for t in amdgpu_targets
        }
        return target_map[(major, minor)]  # type: ignore[index]
    else:
        raise RuntimeError("No GPU is available.")


def match_target(target: Target, target_templates: Sequence[Target]) -> Optional[Target]:
    supported_targets = [tt for tt in target_templates if target.supports(tt)]

    if len(supported_targets) == 0:
        return None

    return max(
        supported_targets, key=lambda tt: (tt.properties.compute_capability[0], tt.properties.compute_capability[1])
    )


def lazy_init():
    # call this function before we use multiprocessing to cache the target
    get_current_target()
