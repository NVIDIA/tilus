"""
Layout inference module for Tilus IR.

Each instruction in the Tilus IR may register a layout inference rule if it allows automatic layout inference.
This module implements a layout inference algorithm that applies these rules to infer the layouts of shared and register
tensors in a given program. It may fail if the rules are missing or if there are conflicting layouts already set in the
program. In such cases, a `LayoutInferenceError` is raised.

Each rule has:
1. A priority that determines the order of application.
2. A validation method that checks if the existing layout of the instruction is compatible with the rule.
3. An inference method that computes the layout for the operands and output tensors that do not have a layout set yet,
   based on the existing layouts.


The layout inference algorithm. We repeat the following steps in a loop:
-----------
1. collect all instructions in the program.
2. check if all shared and register tensors in the program have a layout set. If so, we validate the layouts
2.1 if all instructions with a rule approved the layouts, we return the program.
2.2 if any instruction with a rule rejected the layouts, we raise a `LayoutInferenceError`.
3. sort the instructions by their priority.
4. iterate over the sorted instructions and apply the rules to infer the layouts of the tensors in the instruction.
4.1 if any rule successfully infers a layout for any tensor, we update the program accordingly and go back to step 1.
4.2 if no rule can infer a layout for any tensor, we stop and raise an error indicating inference failure.
-----------
"""

import logging
from typing import Type

from hidet.utils import same_list

from tilus import RegisterLayout, SharedLayout
from tilus.ir import RegisterTensor, SharedTensor
from tilus.ir.func import Analysis, Function
from tilus.ir.inst import Instruction
from tilus.ir.layout.inference.order import rule2order
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceRule,
    get_inference_rules,
    get_validation_rule,
)
from tilus.ir.tools import IRPrinter, collect, rewrite
from tilus.logging import get_logger

logger = get_logger(__name__)


class LayoutInferenceError(Exception):
    pass


def has_missing_layouts(inst: Instruction) -> bool:
    """
    Check if the instruction has any shared or register tensors without a layout set.

    Parameters
    ----------
    inst: Instruction
        The instruction to check.

    Returns
    -------
    bool
        True if the instruction has any shared or register tensors without a layout set, False otherwise.
    """
    operands = inst.inputs + ((inst.output,) if inst.output else ())
    return any(
        isinstance(tensor, (RegisterTensor, SharedTensor)) and tensor.optional_layout is None for tensor in operands
    )


def has_inferred_layouts(inst: Instruction) -> bool:
    """
    Check if the instruction has any shared or register tensors with a layout set.

    Parameters
    ----------
    inst: Instruction
        The instruction to check.

    Returns
    -------
    bool
        True if the instruction has any shared or register tensors with a layout set, False otherwise.
    """
    operands = inst.inputs + ((inst.output,) if inst.output else ())
    return any(
        isinstance(tensor, (RegisterTensor, SharedTensor)) and tensor.optional_layout is not None for tensor in operands
    )


def has_shared_or_register_tensors(inst: Instruction) -> bool:
    """
    Check if the instruction has any shared or register tensors.

    Parameters
    ----------
    inst: Instruction
        The instruction to check.

    Returns
    -------
    bool
        True if the instruction has any shared or register tensors, False otherwise.
    """
    operands = inst.inputs + ((inst.output,) if inst.output else ())
    return any(isinstance(tensor, (RegisterTensor, SharedTensor)) for tensor in operands)


def infer_layout(func: Function) -> Function:
    """
    Infer the layout of the shared and register tensors in the function.

    Raises
    ------
    LayoutInferenceError:
        If the layout inference fails for any reason, such as missing rules or conflicting existed layouts.

    Parameters
    ----------
    func: Function
        The function to infer the layout for.

    Returns
    -------
    ret: Function
        The function with inferred layouts for shared and register tensors. All shared and register tensors in the
        program will have their layouts set.
    """
    logger.debug("Inferring register layouts for function %s", func.name)
    ctx = LayoutInferenceContext(
        num_warps=func.metadata.num_warps,
        num_threads=func.metadata.num_warps * 32,
        analysis=func.metadata.analysis if func.metadata.analysis else Analysis.empty(),
    )
    # step 0: check whether all instructions used in the function have a layout inference rule registered.
    instructions_without_rule = set()
    instructions_without_order = set()
    for inst in collect(func, types=[Instruction]):
        if not has_missing_layouts(inst):
            # if all tensors in the instruction already have a layout set, we skip it
            continue
        rules = get_inference_rules(inst)
        if len(rules) == 0:
            instructions_without_rule.add(inst.__class__)

        for rule in rules:
            if rule not in rule2order:
                instructions_without_order.add(rule)
    if instructions_without_rule or instructions_without_order:
        lines = []
        if instructions_without_rule:
            lines.append(
                "The following instructions do not have a layout inference rule registered: \n"
                + "\n".join("  " + inst.__name__ for inst in instructions_without_rule)
            )
        if instructions_without_order:
            lines.append(
                "The following rules are not specified in the rule order list: \n"
                + "\n".join("  " + rule.__name__ for rule in instructions_without_order)
            )
        raise LayoutInferenceError("\n".join(lines))

    while True:
        all_instructions = collect(func, types=[Instruction])

        # step 1: collect all instructions in the program
        instructions: list[Instruction] = [inst for inst in all_instructions if has_missing_layouts(inst)]

        # step 2: check if all shared and register tensors in the program have a layout set
        if len(instructions) == 0:
            # step 2.1: if all instructions with a rule approved the layouts, we return the program
            invalid_instructions = []
            for inst in all_instructions:
                if has_shared_or_register_tensors(inst) and not get_validation_rule(inst).validate(inst):
                    invalid_instructions.append(inst)
            if not invalid_instructions:
                # all instructions have valid layouts, we can return the function
                return func
            else:
                # step 2.2: if any instruction with a rule rejected the layouts, we raise a `LayoutInferenceError`
                printer = IRPrinter()
                lines = ["The following instructions have invalid layouts: "]
                for inst in invalid_instructions:
                    lines.append(f"  {printer(inst)}")
                for comment, key in printer.comment2key.items():
                    lines.append(f"  {key}: {comment}")
                raise LayoutInferenceError("\n".join(lines))

        # step 3: sort the instructions by their priority
        pairs: list[tuple[Instruction, Type[LayoutInferenceRule]]] = []
        inst2order = {inst: i for i, inst in enumerate(instructions)}
        for inst in instructions:
            for rule in get_inference_rules(inst):
                pairs.append((inst, rule))

        def pair_sort_key(pair: tuple[Instruction, Type[LayoutInferenceRule]]) -> tuple[int, int, int]:
            """
            Sort key for the instruction and rule pair based on the rule's order.
            key:
              no_layout_inferred: instructions with layout inferred should be processed first
              rule_order: rules with lower order should be processed first
              inst_order: instructions appear later in the function should be processed later
            """
            instruction, inference_rule = pair
            return (
                0 if has_inferred_layouts(instruction) else 1,
                rule2order[inference_rule],
                len(instructions) - inst2order[instruction],
            )

        pairs.sort(key=pair_sort_key)
        # print("Sorted instruction and rule pairs:")
        # for inst, rule in pairs:
        #     print(f"  {inst} with rule {rule.__name__} ({pair_sort_key((inst, rule))})")
        # print()

        # step 4: iterate over the sorted instructions and apply the rules to infer the layouts of the tensors
        found = False
        for inst, rule in pairs:
            # skip the instruction if all its shared and register tensors already have a layout set
            if all(
                tensor.optional_layout is not None
                for tensor in inst.inputs + (inst.output,)
                if isinstance(tensor, (RegisterTensor, SharedTensor))
            ):
                continue

            mapping = rule.inference(ctx, inst)
            if mapping:
                # step 4.1: if any rule successfully infers a layout for any tensor, we update the program
                rewrite_map: dict[SharedTensor | RegisterTensor, SharedTensor | RegisterTensor] = {}
                for tensor, layout in mapping.items():
                    assert same_list(tensor.shape, layout.shape), (
                        f"Layout shape does not match tensor shape: {tensor.shape} vs {layout.shape} for rule {rule.__name__} "
                    )
                    if isinstance(tensor, RegisterTensor) and isinstance(layout, RegisterLayout):
                        rewrite_map[tensor] = tensor.with_layout(layout)
                    elif isinstance(tensor, SharedTensor) and isinstance(layout, SharedLayout):
                        rewrite_map[tensor] = tensor.with_layout(layout)
                    else:
                        raise TypeError(f"Invalid layout type {type(layout)} for tensor {tensor}. ")
                func = rewrite(func, rewrite_map)
                if logger.isEnabledFor(logging.DEBUG):
                    printer = IRPrinter()
                    logger.debug("%s", str(printer(inst)))
                    for tensor, layout in mapping.items():
                        logger.debug("  %s: %s", str(printer(tensor)), layout)
                found = True
                break
        if not found:
            # step 4.2: if no rule can infer a layout for any tensor, we stop and raise an error indicating
            # inference failure
            printer = IRPrinter()
            func_doc = printer(func)

            tensors = set()
            for inst in instructions:
                for tensor in inst.inputs + ((inst.output,) if inst.output else ()):
                    if isinstance(tensor, (RegisterTensor, SharedTensor)) and tensor.optional_layout is None:
                        tensors.add(tensor)
            lines = [
                "Layout inference failed: no rules could infer a layout for the following tensors:",
            ]
            for tensor in tensors:
                lines.append(f"  {printer(tensor)}: {tensor.dtype.name}{list(tensor.shape)}")
            lines.append(f"\nFunction:\n{func_doc}")
            raise LayoutInferenceError("\n".join(lines))
