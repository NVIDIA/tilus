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
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.instructions import AnnotateLayoutInst
from tilus.ir.layout.inference import infer_layout, verify_layouts
from tilus.ir.layout.inference.inference import LayoutInferenceError
from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.ir.tools import IRPrinter, rewrite
from tilus.transforms.base import Pass


class ApplyLayoutAnnotationRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.remap = {}

    def visit_AnnotateLayoutInst(self, inst: AnnotateLayoutInst) -> None:
        tensor = inst.register_input
        layout = inst.layout

        if tensor.optional_layout is not None and tensor.optional_layout != layout:
            raise ValueError(
                f"Tensor {tensor} already has a different layout {tensor.optional_layout}, cannot annotate with {layout}."
            )

        if tensor in self.remap:
            existing_tensor = self.remap[tensor]
            if existing_tensor.layout != layout:
                raise ValueError(
                    f"Tensor {tensor} already remapped to {existing_tensor}, cannot annotate with {layout}."
                )
            return

        self.remap[tensor] = tensor.with_layout(layout)

    def visit_Function(self, func: Function) -> Function:
        self.remap.clear()
        updated_func = super().visit_Function(func)

        if updated_func is func and not self.remap:
            return func

        updated_func = rewrite(updated_func, self.remap)
        return updated_func


class LayoutInferencePass(Pass):
    def process_function(self, func: Function) -> Function:
        apply_annotation = ApplyLayoutAnnotationRewriter()
        func = apply_annotation(func)
        func = infer_layout(func)
        self.verify_layouts(func)

        return func

    @staticmethod
    def verify_layouts(func):
        invalid_instructions = verify_layouts(func)

        if invalid_instructions:
            printer = IRPrinter()
            program_text = str(printer(func).indent())
            lines = [
                "Layout verification failed for the following program:",
                program_text,
                "",
                "Instructions with invalid layouts:",
            ]
            for inst in invalid_instructions:
                lines.append(str(printer(inst)))
                tensors = inst.inputs + ((inst.output,) if inst.output else tuple())
                for tensor in tensors:
                    if isinstance(tensor, (SharedTensor, RegisterTensor)):
                        lines.append("  " + str(printer(tensor)) + ": " + str(tensor.optional_layout))
                lines.append("")
            error_message = "\n".join(lines)
            raise LayoutInferenceError(error_message)
        pass


def layout_inference_pass() -> Pass:
    return LayoutInferencePass()
