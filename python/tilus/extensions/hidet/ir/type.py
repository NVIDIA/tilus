from hidet.ir.type import BaseType, DataType, FuncType, PointerType, TensorPointerType, TensorType, VoidType


def sizeof(tp: BaseType) -> int:
    from hidet.utils import prod

    if isinstance(tp, DataType):
        return tp.nbytes
    elif isinstance(tp, (PointerType, TensorPointerType)):
        return 8
    elif isinstance(tp, TensorType):
        return sizeof(tp.dtype) * prod(tp.shape)
    else:
        raise NotImplementedError(type(tp))


def type_equal(lhs: BaseType, rhs: BaseType) -> bool:
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, DataType) and isinstance(rhs, DataType):
        return lhs.name == rhs.name
    elif isinstance(lhs, PointerType) and isinstance(rhs, PointerType):
        return type_equal(lhs.base_type, rhs.base_type)
    elif isinstance(lhs, VoidType) and isinstance(rhs, VoidType):
        return True
    elif isinstance(lhs, TensorPointerType) and isinstance(rhs, TensorPointerType):
        return type_equal(lhs.tensor_type, rhs.tensor_type)
    elif isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        from hidet.ir.expr import is_constant

        if not type_equal(lhs.dtype, rhs.dtype):
            return False
        if len(lhs.shape) != len(rhs.shape):
            return False
        for a, b in zip(lhs.shape, rhs.shape):
            if is_constant(a) ^ is_constant(b):
                return False
            elif is_constant(a) and is_constant(b):
                if int(a) != int(b):
                    return False
            else:
                # we do not have equivalence checking for symbolic expression
                pass
        # do not check layout
        return True
    elif isinstance(lhs, FuncType) and isinstance(rhs, FuncType):
        assert lhs.param_types is not None and lhs.ret_type is not None
        assert rhs.param_types is not None and rhs.ret_type is not None
        if len(lhs.param_types) != len(rhs.param_types):
            return False
        if not type_equal(lhs.ret_type, rhs.ret_type):
            return False
        for a, b in zip(lhs.param_types, rhs.param_types):
            if not type_equal(a, b):
                return False
        return True
    else:
        raise NotImplementedError()


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
