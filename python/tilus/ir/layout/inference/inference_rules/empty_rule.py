from tilus import RegisterLayout, SharedLayout
from tilus.ir.instructions import (
    AllocateRegisterInst,
    AllocateSharedInst,
    FormatPrintInst,
    FreeSharedInst,
    GlobalViewInst,
    PrintTensorInst,
    StoreGlobalInst,
)
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import Tensor


@register_rule(PrintTensorInst)
@register_rule(FormatPrintInst)
@register_rule(FreeSharedInst)
@register_rule(AllocateRegisterInst)
@register_rule(AllocateSharedInst)
@register_rule(StoreGlobalInst)
class EmptyRule(LayoutInferenceRule):
    @staticmethod
    def validate(inst: GlobalViewInst) -> bool:
        return True

    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: GlobalViewInst) -> dict[Tensor, SharedLayout | RegisterLayout]:
        return {}
