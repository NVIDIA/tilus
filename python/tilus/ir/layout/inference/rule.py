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
from collections import defaultdict
from typing import Callable, Type, TypeVar

from hidet.utils.py import bold

from tilus.ir.func import Analysis
from tilus.ir.inst import Instruction
from tilus.ir.layout import RegisterLayout, SharedLayout
from tilus.ir.tensor import RegisterTensor, SharedTensor


class LayoutInferenceError(Exception):
    pass


class LayoutInferenceContext:
    def __init__(self, num_threads: int, thread_begin: int, thread_end: int, analysis: Analysis):
        self.num_threads: int = num_threads
        self.thread_begin: int = thread_begin
        self.thread_end: int = thread_end
        self.analysis: Analysis = analysis


class LayoutValidationRule:
    @staticmethod
    def validate(inst: Instruction) -> bool:
        """
        Validate if the instruction is valid for the given layout.

        Parameters
        ----------
        inst: Instruction
            The instruction to validate.

        Returns
        -------
        bool
            True if the instruction is valid for the given layout, False otherwise.
        """
        raise NotImplementedError()


class LayoutInferenceRule:
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: Instruction
    ) -> dict[RegisterTensor | SharedTensor, RegisterLayout | SharedLayout]:
        """
        Perform layout inference for the register and shared tensors in the given instruction.

        This method tries to infer the layout for the tensors in the instruction that do not have a layout set yet,
        based on the existing layouts of the tensors. If the instruction has no tensors without a layout set,
        it may determine the layouts with a predefined logic according to the dtype and shape of the tensors, as well
        as the instruction arguments.

        It may only infer the layouts for part of the tensors in the instruction.

        The tensors with inferred layouts are returned as a mapping from the original tensor to the inferred tensor.

        If it can not infer the layout for any tensor, it should return an empty mapping.

        Parameters
        ----------
        ctx: LayoutInferenceContext
            The context for layout inference, containing information might be needed for inference.

        inst: Instruction
            The instruction to infer the layout for.

        Returns
        -------
        mapping: dict[RegisterTensor | SharedTensor, RegisterLayout | SharedLayout]
            The mapping from the tensor to the inferred layout.
        """
        raise NotImplementedError()


# each instruction type can have multiple layout inference rules registered
_inference_rules: dict[Type[Instruction], list[Type[LayoutInferenceRule]]] = defaultdict(list)

# each instruction type must have one validation rule registered
_validation_rules: dict[Type[Instruction], Type[LayoutValidationRule]] = {}

InstClassT = TypeVar("InstClassT", bound=Type[Instruction])
RuleClassT = TypeVar("RuleClassT", bound=Type[LayoutInferenceRule | LayoutValidationRule])


def register_rule(inst_type: InstClassT) -> Callable[[RuleClassT], RuleClassT]:
    def decorator(rule_class: RuleClassT) -> RuleClassT:
        if issubclass(rule_class, LayoutValidationRule):
            if inst_type in _validation_rules:
                raise ValueError(f"Validation rule for {inst_type.__name__} is already registered")
            _validation_rules[inst_type] = rule_class
        elif issubclass(rule_class, LayoutInferenceRule):
            _inference_rules[inst_type].append(rule_class)
        else:
            raise TypeError(f"{rule_class.__name__} is not a subclass of LayoutValidationRule or LayoutInferenceRule")
        return rule_class

    return decorator


def get_inference_rules(inst: Type[Instruction] | Instruction) -> list[Type[LayoutInferenceRule]]:
    """Get the layout inference rule for the given instruction class."""
    if isinstance(inst, Instruction):
        inst_cls = type(inst)
    elif isinstance(inst, type) and issubclass(inst, Instruction):
        inst_cls = inst
    else:
        raise TypeError(f"Expected an Instruction instance or a subclass of Instruction, got {type(inst)}")

    if inst_cls not in _inference_rules:
        for parent_cls in inst_cls.__mro__:
            if parent_cls in _inference_rules:
                _inference_rules[inst_cls] = _inference_rules[parent_cls]
                break
        else:
            return []
    return _inference_rules[inst_cls].copy()


def get_validation_rule(inst: Type[Instruction] | Instruction) -> Type[LayoutValidationRule]:
    """Get the layout validation rule for the given instruction class."""
    if isinstance(inst, Instruction):
        inst_cls = type(inst)
    elif isinstance(inst, type) and issubclass(inst, Instruction):
        inst_cls = inst
    else:
        raise TypeError(f"Expected an Instruction instance or a subclass of Instruction, got {type(inst)}")

    if inst_cls not in _validation_rules:
        for parent_cls in inst_cls.__mro__:
            if parent_cls in _validation_rules:
                _validation_rules[inst_cls] = _validation_rules[parent_cls]
                break
        raise ValueError(f"No layout validation rule registered for {bold(inst_cls.__name__)}")

    return _validation_rules[inst_cls]
