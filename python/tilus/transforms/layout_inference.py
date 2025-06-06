from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.instructions import AnnotateLayoutInst
from tilus.ir.layout.inference import infer_layout
from tilus.ir.tools import rewrite
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
        return func


def layout_inference_pass() -> Pass:
    return LayoutInferencePass()
