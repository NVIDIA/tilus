from tilus.ir.instructions import (
    AllocateRegisterInst,
    AllocateSharedInst,
    CopyAsyncInst,
    FormatPrintInst,
    FreeSharedInst,
    Instruction,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedGenericInst,
    LoadSharedInst,
    PrintTensorInst,
    SharedSliceInst,
    StoreGlobalGenericInst,
    StoreGlobalInst,
    StoreSharedInst,
)
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule


@register_rule(StoreGlobalGenericInst)
@register_rule(PrintTensorInst)
@register_rule(FormatPrintInst)
@register_rule(LoadGlobalInst)
@register_rule(LoadGlobalGenericInst)
@register_rule(StoreGlobalInst)
@register_rule(SharedSliceInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
@register_rule(FreeSharedInst)
@register_rule(AllocateRegisterInst)
@register_rule(AllocateSharedInst)
class AlwaysOkayRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        return True


# todo: the following instructions should have dedicated validation rules
@register_rule(StoreSharedInst)
@register_rule(CopyAsyncInst)
class TemporaryAlwaysOkRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        return True
