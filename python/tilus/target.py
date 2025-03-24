import functools
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class TargetProperties:
    compute_capability: Tuple[int, int] = (0, 0)
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
        assert isinstance(target, Target)
        # whether the features supported by the self target contains features in the target
        if target == gpgpu_any:
            return True
        if self.kind != target.kind:
            return False
        return self.properties.compute_capability >= target.properties.compute_capability


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
amdgpu_any = Target(kind="amdgpu", arch="any", properties=TargetProperties())
amdgpu_gfx1100 = Target(  # e.g., RX 7900 XTX
    kind="amdgpu",
    arch="gfx1100",
    properties=TargetProperties(compute_capability=(11, 0), shared_memory_per_block=64 * 1024),
)

"""
    NVIDIA GPUs
    
    See Also: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities
"""
nvgpu_any = Target("nvgpu", "any", TargetProperties())
nvgpu_sm70 = Target("nvgpu", "sm70", TargetProperties(compute_capability=(7, 0), shared_memory_per_block=96 * 1024))
nvgpu_sm75 = Target("nvgpu", "sm75", TargetProperties(compute_capability=(7, 5), shared_memory_per_block=96 * 1024))
nvgpu_sm80 = Target("nvgpu", "sm80", TargetProperties(compute_capability=(8, 0), shared_memory_per_block=163 * 1024))
nvgpu_sm86 = Target("nvgpu", "sm86", TargetProperties(compute_capability=(8, 6), shared_memory_per_block=99 * 1024))
nvgpu_sm89 = Target("nvgpu", "sm89", TargetProperties(compute_capability=(8, 9), shared_memory_per_block=99 * 1024))
nvgpu_sm90 = Target("nvgpu", "sm90", TargetProperties(compute_capability=(9, 0), shared_memory_per_block=227 * 1024))


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
        nvgpu_targets = [nvgpu_sm70, nvgpu_sm75, nvgpu_sm80, nvgpu_sm86, nvgpu_sm89, nvgpu_sm90]
        target_map = {t.properties.compute_capability: t for t in nvgpu_targets}
        return target_map[(major, minor)]
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
        target_map = {t.properties.compute_capability: t for t in amdgpu_targets}
        return target_map[(major, minor)]
    else:
        raise RuntimeError("No GPU is available.")


def match_target(target: Target, target_templates: Sequence[Target]) -> Optional[Target]:
    supported_targets = [tt for tt in target_templates if target.supports(tt)]

    if len(supported_targets) == 0:
        return None

    return max(supported_targets, key=lambda tt: tt.properties.compute_capability)


# cache the result in case the sub-process is forked and call it again, which will cause cuda initialization error
get_current_target()
