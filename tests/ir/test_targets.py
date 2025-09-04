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
"""
Test target support logic with suffix semantics.
"""

import pytest
from tilus.target import Target, TargetProperties, gpgpu_any


def create_target(arch, major, minor, suffix=None):
    """Helper to create test targets"""
    return Target(
        kind="nvgpu",
        arch=arch,
        properties=TargetProperties(
            compute_capability=(major, minor), feature_suffix=suffix, shared_memory_per_block=100 * 1024
        ),
    )


# Create test targets as fixtures for reuse
@pytest.fixture(scope="module")
def test_targets():
    """Create all test targets"""
    return {
        "sm80": create_target("sm80", 8, 0, None),  # Base sm80
        "sm90": create_target("sm90", 9, 0, None),  # Base sm90
        "sm90a": create_target("sm90a", 9, 0, "a"),  # Architecture-specific sm90
        "sm90f": create_target("sm90f", 9, 0, "f"),  # Family-specific sm90
        "sm100": create_target("sm100", 10, 0, None),  # Base sm100
        "sm100a": create_target("sm100a", 10, 0, "a"),  # Architecture-specific sm100
        "sm100f": create_target("sm100f", 10, 0, "f"),  # Family-specific sm100
        "sm103": create_target("sm103", 10, 3, None),  # Base sm103
        "sm103f": create_target("sm103f", 10, 3, "f"),  # Family-specific sm103
        "sm103a": create_target("sm103a", 10, 3, "a"),  # Architecture-specific sm103
        "sm110": create_target("sm110", 11, 0, None),  # Base sm110
        "sm110a": create_target("sm110a", 11, 0, "a"),  # Architecture-specific sm110
        "sm120": create_target("sm120", 12, 0, None),  # Base sm120
    }


class TestTargetSupports:
    """Test cases for Target.supports() method with suffix semantics"""

    @pytest.mark.parametrize(
        "self_target,target_to_check,expected",
        [
            # Basic capability tests
            ("sm90", "sm80", True),
            ("sm80", "sm90", False),
            ("sm110", "sm100", True),
            ("sm100", "sm110", False),
        ],
        ids=[
            "sm90_supports_sm80_base_capabilities",
            "sm80_does_not_support_sm90_insufficient_capability",
            "sm110_supports_sm100_higher_major",
            "sm100_does_not_support_sm110_lower_major",
        ],
    )
    def test_basic_capability_support(self, test_targets, self_target, target_to_check, expected):
        """Test basic compute capability comparisons"""
        result = test_targets[self_target].supports(test_targets[target_to_check])
        assert result == expected

    def test_gpgpu_any_support(self, test_targets):
        """Test that any target supports gpgpu_any"""
        assert test_targets["sm90"].supports(gpgpu_any)
        assert test_targets["sm80"].supports(gpgpu_any)
        assert test_targets["sm110a"].supports(gpgpu_any)

    @pytest.mark.parametrize(
        "self_target,target_to_check,expected",
        [
            # Base target supporting/not supporting suffixed targets
            ("sm90", "sm90", True),
            ("sm90", "sm90a", False),
            ("sm90", "sm90f", False),
            ("sm100", "sm100a", False),
            ("sm100", "sm100f", False),
            ("sm110", "sm103f", False),
        ],
        ids=[
            "sm90_supports_sm90_same_base",
            "sm90_base_does_not_support_sm90a_architecture_specific",
            "sm90_base_does_not_support_sm90f_family_specific",
            "sm100_base_does_not_support_sm100a_architecture_specific",
            "sm100_base_does_not_support_sm100f_family_specific",
            "sm110_base_does_not_support_sm103f_family_specific",
        ],
    )
    def test_base_target_suffix_support(self, test_targets, self_target, target_to_check, expected):
        """Test base targets supporting/not supporting suffixed targets"""
        result = test_targets[self_target].supports(test_targets[target_to_check])
        assert result == expected

    @pytest.mark.parametrize(
        "self_target,target_to_check,expected",
        [
            # Architecture-specific target tests
            ("sm90a", "sm90", True),
            ("sm90a", "sm90a", True),
            ("sm90a", "sm90f", True),
            ("sm90a", "sm100a", False),
            ("sm110a", "sm90a", False),
            ("sm103a", "sm103f", True),
            ("sm110a", "sm100a", False),
        ],
        ids=[
            "sm90a_supports_sm90_base",
            "sm90a_supports_sm90a_same_architecture",
            "sm90a_supports_sm90f_family_same_version",
            "sm90a_does_not_support_sm100a_different_version",
            "sm110a_does_not_support_sm90a_architecture_specific_not_backward_compatible",
            "sm103a_supports_sm103f_architecture_includes_family_same_version",
            "sm110a_does_not_support_sm100a_different_architectures",
        ],
    )
    def test_architecture_specific_support(self, test_targets, self_target, target_to_check, expected):
        """Test architecture-specific target support rules"""
        result = test_targets[self_target].supports(test_targets[target_to_check])
        assert result == expected

    @pytest.mark.parametrize(
        "self_target,target_to_check,expected",
        [
            # Family-specific target tests
            ("sm100f", "sm100", True),
            ("sm100f", "sm100f", True),
            ("sm100f", "sm100a", False),
            ("sm103f", "sm100f", True),
            ("sm100f", "sm103f", False),
        ],
        ids=[
            "sm100f_supports_sm100_base",
            "sm100f_supports_sm100f_same_family",
            "sm100f_does_not_support_sm100a_architecture_specific",
            "sm103f_supports_sm100f_later_minor_in_same_major",
            "sm100f_does_not_support_sm103f_earlier_minor",
        ],
    )
    def test_family_specific_support(self, test_targets, self_target, target_to_check, expected):
        """Test family-specific target support rules"""
        result = test_targets[self_target].supports(test_targets[target_to_check])
        assert result == expected

    @pytest.mark.parametrize(
        "self_target,target_to_check,expected",
        [
            # Cross-major version tests
            ("sm110", "sm100f", False),
            ("sm120", "sm110a", False),
        ],
        ids=[
            "sm110_base_does_not_support_sm100f_family_specific",
            "sm120_base_does_not_support_sm110a_architecture_specific",
        ],
    )
    def test_cross_major_version_support(self, test_targets, self_target, target_to_check, expected):
        """Test support rules across different major versions"""
        result = test_targets[self_target].supports(test_targets[target_to_check])
        assert result == expected


class TestTargetProperties:
    """Test Target and TargetProperties classes"""

    def test_target_creation(self):
        """Test basic target creation"""
        target = create_target("sm90a", 9, 0, "a")
        assert target.kind == "nvgpu"
        assert target.arch == "sm90a"
        assert target.properties.compute_capability == (9, 0)
        assert target.properties.feature_suffix == "a"
        assert target.properties.shared_memory_per_block == 100 * 1024

    def test_target_string_representation(self):
        """Test target string representation"""
        target = create_target("sm90a", 9, 0, "a")
        assert str(target) == "nvgpu/sm90a"

    def test_target_kind_checks(self):
        """Test target kind checking methods"""
        nvgpu_target = create_target("sm90", 9, 0, None)
        amdgpu_target = Target(
            kind="amdgpu",
            arch="gfx1100",
            properties=TargetProperties(compute_capability=(11, 0), shared_memory_per_block=64 * 1024),
        )

        assert nvgpu_target.is_nvgpu()
        assert not nvgpu_target.is_amdgpu()
        assert not amdgpu_target.is_nvgpu()
        assert amdgpu_target.is_amdgpu()

    @pytest.mark.parametrize(
        "major,minor,suffix,expected_capability",
        [
            (7, 5, None, (7, 5)),
            (9, 0, "a", (9, 0)),
            (10, 3, "f", (10, 3)),
            (12, 1, "a", (12, 1)),
        ],
    )
    def test_compute_capability_format(self, major, minor, suffix, expected_capability):
        """Test that compute capability is stored in correct format"""
        target = create_target(f"sm{major}{minor}", major, minor, suffix)
        assert target.properties.compute_capability == expected_capability


class TestTargetCompatibility:
    """Test target compatibility edge cases"""

    def test_same_target_supports_itself(self, test_targets):
        """Test that any target supports itself"""
        for target_name, target in test_targets.items():
            assert target.supports(target), f"{target_name} should support itself"

    def test_different_kinds_not_supported(self):
        """Test that targets of different kinds don't support each other"""
        nvgpu_target = create_target("sm90", 9, 0, None)
        amdgpu_target = Target(
            kind="amdgpu",
            arch="gfx1100",
            properties=TargetProperties(compute_capability=(11, 0), shared_memory_per_block=64 * 1024),
        )

        assert not nvgpu_target.supports(amdgpu_target)
        assert not amdgpu_target.supports(nvgpu_target)
