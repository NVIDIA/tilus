from tilus.ir.func import Function
from tilus.ir.layout.inference import infer_layout
from tilus.transforms.base import Pass


class LayoutInferencePass(Pass):
    def process_function(self, func: Function) -> Function:
        return infer_layout(func)


def layout_inference_pass() -> Pass:
    return LayoutInferencePass()
