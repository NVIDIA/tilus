from tilus import RegisterLayout
from tilus.ir.instructions import CopyAsyncGenericInst, CopyAsyncInst
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(CopyAsyncInst)
@register_rule(CopyAsyncGenericInst)
class CopyAsyncRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: CopyAsyncGenericInst | CopyAsyncInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        raise NotImplementedError()
