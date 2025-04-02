from hidet.ir.type import BaseType, PointerType, TensorPointerType, TensorType


def is_addressable(tp_or_var):
    from hidet.ir.expr import Var

    if isinstance(tp_or_var, Var):
        tp = tp_or_var.type
    else:
        tp = tp_or_var
    return isinstance(tp, (PointerType, TensorPointerType, TensorType))


def get_base_type(tp: BaseType) -> BaseType:
    if isinstance(tp, PointerType):
        return tp.base_type
    elif isinstance(tp, TensorPointerType):
        return tp.tensor_type.dtype
    elif isinstance(tp, TensorType):
        return tp.dtype
    else:
        assert False
