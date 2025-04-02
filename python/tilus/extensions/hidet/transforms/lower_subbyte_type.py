from typing import Dict, List, Tuple, Union

import tabulate

from hidet.ir import Add, AssignStmt, BufferStoreStmt, Cast, DeclareStmt, LetStmt, Sub
from hidet.ir.dtypes import boolean, float32, int32, int64, uint8, uint16, uint32
from hidet.ir.dtypes.integer_subbyte import IntegerSubbyteType
from hidet.ir.expr import Address, Call, Constant, Dereference, Expr, SymbolVar, TensorElement, Var, cast, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.layout import row_major
from hidet.ir.primitives.cuda.atomic import atomic_cas
from hidet.ir.stmt import DeclareScope, Stmt
from hidet.ir.tools import IRPrinter, TypeInfer, collect
from hidet.ir.type import BaseType, DataType, PointerType, TensorPointerType, TensorType, type_equal
from hidet.transforms.base import FunctionPass
from hidet.utils import same_list
from tilus.extensions.hidet.ir.builders import StmtBuilder
from tilus.extensions.hidet.ir.primitives.cuda.cast import cast_subbyte_float_from_f32, cast_subbyte_float_to_f32
from tilus.extensions.hidet.ir.primitives.cuda.subbyte import load_subbyte
from tilus.extensions.hidet.ir.type import get_base_type, is_addressable
from tilus.target import get_current_target


class Addressing:
    """
    Addressable expression: an expression that points to a memory location.

    It will have one of the following types:
    - pointer type (PointerType)
    - tensor pointer type (TensorPointerType)
    - tensor (TensorType)

    Examples:
    - a = register_tensor(dtype='float16', shape=[10, 10])
    - &a[0, 1]
    - &a[0, 1] + 1
    - &b (where b is an int32 in register)

    For each addressable expression, we will analyze its
    - scope: the scope where the memory is allocated, like global memory, shared memory, register
    - type: the element type of the pointer points to
    - base: the base expression of the pointer
    - offset: the offset from the base expression

    For example, for the expression &a[0, 1] + 1, we will have:
    - scope: DeclareScope.Register
    - type: float16
    - base: &a[0, 1]
    - offset: 1
    """

    def __init__(self, scope: DeclareScope, type: BaseType, base: Expr, offset: Expr):
        self.scope: DeclareScope = scope
        self.type: BaseType = type  # base type of pointer
        self.base: Expr = base
        self.offset: Expr = offset


def _get_type(tp_or_var: Union[Var, BaseType]) -> BaseType:
    if isinstance(tp_or_var, Var):
        return tp_or_var.type
    else:
        return tp_or_var


def is_pointer(tp_or_var: Union[Var, BaseType]) -> bool:
    tp = _get_type(tp_or_var)
    return isinstance(tp, (PointerType, TensorPointerType))


def is_integer(tp_or_var: Union[Var, BaseType]) -> bool:
    tp = _get_type(tp_or_var)
    return isinstance(tp, DataType) and tp.is_integer()


def is_subbyte(tp_or_var: Union[Var, BaseType]) -> bool:
    tp = _get_type(tp_or_var)
    return isinstance(tp, DataType) and tp.nbits < 8


class AddressingAnalyzer(IRVisitor):
    """
    Analyzer that analyzes the addressing information of all addressable expressions.
    """

    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()
        self.printer = IRPrinter()
        self.buf2addr: Dict[Expr, Addressing] = {}

    def debug_analyze_and_print(self, func: Function) -> None:
        printer = IRPrinter()
        self.visit(func)
        headers = ["scope", "dtype", "buf", "base", "offset"]
        rows = []
        for buf, addr in self.buf2addr.items():
            rows.append([addr.scope.name, addr.type, printer(buf), printer(addr.base), printer(addr.offset)])

        print(printer(func))
        print(tabulate.tabulate(rows, headers=headers))
        print()

    def visit(self, node):
        ret = super().visit(node)

        # perform a post-visit check to make sure we get the addressing information for all addressable expressions
        if isinstance(node, Expr):
            ret_type = self.type_infer(node)
            if is_addressable(ret_type) and is_subbyte(get_base_type(ret_type)):
                assert node in self.buf2addr, self.printer(node)

        return ret

    def visit_Function(self, func: Function) -> Function:
        for param in func.params:
            if is_addressable(param):
                self.buf2addr[param] = Addressing(
                    base=param, type=get_base_type(param.type), scope=DeclareScope.Global, offset=int32.zero
                )

        symbol_vars = collect(func.body, SymbolVar)
        for symbol_var in symbol_vars:
            if is_addressable(symbol_var):
                self.buf2addr[symbol_var] = Addressing(
                    base=symbol_var, type=get_base_type(symbol_var.type), scope=DeclareScope.Global, offset=int32.zero
                )

        return super().visit_Function(func)

    def visit_Cast(self, e: Cast) -> Expr:
        self.visit(e.expr)
        if is_addressable(e.target_type) and e.expr in self.buf2addr:
            addr = self.buf2addr[e.expr]
            self.buf2addr[e] = Addressing(
                scope=addr.scope, type=get_base_type(e.target_type), base=e, offset=int32.zero
            )
        return super().visit_Cast(e)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Stmt:
        if is_addressable(stmt.var):
            if isinstance(stmt.var.type, TensorType):
                # defined a tensor, we know its scope from the declare statement
                if stmt.scope == DeclareScope.Default:
                    scope = DeclareScope.Register
                else:
                    scope = stmt.scope
            else:
                # pointer or tensor pointer, we need to check the init value to know its scope
                if stmt.init is not None and stmt.init in self.buf2addr:
                    self.visit(stmt.init)
                    scope = self.buf2addr[stmt.init].scope
                else:
                    # unknown for now, will be updated in AssignStmt
                    scope = DeclareScope.Default
            self.buf2addr[stmt.var] = Addressing(
                scope=scope, type=get_base_type(stmt.var.type), base=stmt.var, offset=int32.zero
            )

        return super().visit_DeclareStmt(stmt)

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        if is_addressable(stmt.var):
            self.visit(stmt.value)
            value_addr = self.buf2addr[stmt.value]
            if self.buf2addr[stmt.var].scope == DeclareScope.Default:
                self.buf2addr[stmt.var].scope = value_addr.scope
            else:
                # we do not allow a pointer that can points to different scopes at different time
                # check consistency
                assert value_addr.scope == DeclareScope.Default or self.buf2addr[stmt.var].scope == value_addr.scope, (
                    "Inconsistent scope for variable {}: {} vs {}".format(
                        stmt.var.name, self.buf2addr[stmt.var].scope, value_addr.scope
                    )
                )
        return super().visit_AssignStmt(stmt)

    def visit_Call(self, e: Call) -> Expr:
        if is_addressable(self.type_infer(e)):
            # this is a function returns a pointer, we will handle case by case for different functions
            func_name = e.func_var.name
            if "request_cuda_workspace" in func_name:
                scope = DeclareScope.Global
            elif "get_cuda_stream" in func_name:
                scope = DeclareScope.Host
            elif "dynamic_shared_memory" in func_name:
                scope = DeclareScope.Shared
            else:
                raise ValueError("Can not infer the scope of return value of function {}".format(func_name))

            self.buf2addr[e] = Addressing(
                scope=scope, type=get_base_type(self.type_infer(e)), base=e, offset=int32.zero
            )
        return super().visit_Call(e)

    def visit_Address(self, e: Address) -> Expr:
        super().visit(e.expr)
        if isinstance(e.expr, TensorElement):
            scope = self.buf2addr[e.expr.base].scope
        elif isinstance(e.expr, Dereference):
            scope = self.buf2addr[e.expr.expr].scope
        elif isinstance(e.expr, Var):
            var: Var = e.expr
            if isinstance(var.type, DataType):
                scope = DeclareScope.Register
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(str(type(e.expr)) + " " + str(e))
        self.buf2addr[e] = Addressing(scope=scope, type=get_base_type(self.type_infer(e)), base=e, offset=int32.zero)
        return super().visit_Address(e)

    def visit_binary(self, e: Expr, lhs: Expr, rhs: Expr, op: str) -> None:
        assert op in ["+", "-"]
        if is_addressable(self.type_infer(e)):
            op_f = {"+": lambda a, b: a + b, "-": lambda a, b: a - b}[op]
            self.visit(lhs)
            self.visit(rhs)
            lhs_type = self.type_infer(lhs)
            rhs_type = self.type_infer(rhs)
            if is_pointer(lhs_type) and is_integer(rhs_type) and op in ["+", "-"]:
                lhs_addr = self.buf2addr[lhs]
                self.buf2addr[e] = Addressing(
                    scope=lhs_addr.scope, type=lhs_addr.type, base=lhs_addr.base, offset=op_f(lhs_addr.offset, rhs)
                )
            elif is_integer(lhs_type) and is_pointer(rhs_type) and op in ["+"]:
                rhs_addr = self.buf2addr[rhs]
                self.buf2addr[e] = Addressing(
                    scope=rhs_addr.scope, type=rhs_addr.type, base=rhs_addr.base, offset=op_f(lhs, rhs_addr.offset)
                )
            else:
                raise ValueError()

    def visit_Add(self, e: Add) -> Expr:
        self.visit_binary(e, e.a, e.b, "+")
        return super().visit_Add(e)

    def visit_Sub(self, e: Sub) -> Expr:
        self.visit_binary(e, e.a, e.b, "-")
        return super().visit_Sub(e)


class LowerSubbyteTypeRewriter(IRRewriter):
    """
    Lower all sub-byte types to uint8 data type.

    Please note that this pass serves as the fall-back implementation for the sub-byte type when there is no
    other optimizations performed before to eliminate the sub-byte type access.

    We assume all high-rank tensors have been flattened to 1-dimension.

    Transformations that lower the sub-byte type to uint8 type:

    1. variable definitions
        1.1 define a tensor with sub-byte data type:
               a = register_tensor('uint4b', shape=[n])
            => a = register_tensor('uint8', shape=[(n + 1) // 2])
        1.3 define a scalar variable with sub-byte data type:
               a = uint4b(1)
            => a = uint8(1)
        1.2 define a pointer that points to a sub-byte data type:
               t = register_tensor('uint3b', shape=[n])
               a = ~t[3]
            => t = register_tensor('uint8', shape=[(n * 3 + 7) // 8])
               bit_index = 3 * 3
               a = ~t[bit_index // 8]
               a_bit_offset = bit_index % 8
            where we used an extra variable to track the bit offset
        1.3 define a tensor pointer with sub-byte data type:
            same as case 1.2 since tensor pointer is also a pointer that points to the first element of the tensor.

    2. element access:
        2.1 tensor element access
        2.2 tensor pointer element access
        2.3 pointer access
        all transformed to access the bits and returns an uint8 value where the low bits hold the value of the sub-byte
        type.

    3. buffer store:
        buf_expr[index_expr] = value_expr
        will be transformed to statements that update the corresponding bits in the buffer.

    4. pointer arithmetics:
        Any expression with sub-byte type pointer will be transformed to two expressions:
        1. uint8_pointer: ~uint8 - represents the pointer to the byte where the first bit of the sub-byte value is stored.
        2. bit_offset: int32 - the bit-offset of the first bit of the sub-byte value in above byte.

    5. sub-byte type arithmatic operations are performed with int32 and float32 as the proxy data type
        5.1 a_int4 + b_int4 => (int4)((int32)a_int4 + (int32)b_int4)
        5.2 a_float5 + b_float5 => (float5)((float32)a_float5 + (float32)b_float5)
    """

    def __init__(self):
        super().__init__()
        self.analyzer = AddressingAnalyzer()
        self.type_infer = self.analyzer.type_infer
        self.buf2addr: Dict[Expr, Addressing] = self.analyzer.buf2addr
        self.uint8_pointer: Dict[Var, Expr] = {}
        self.bit_offset: Dict[Var, Expr] = {}

    def get_byte_and_bit_offset(self, expr: Expr) -> Tuple[Expr, Expr]:
        """
        Get the byte and bit-offset of the pointer expression in the buffer.
        """
        expr_type = self.type_infer(expr)
        dtype = get_base_type(expr_type)
        assert is_addressable(expr_type)
        assert is_subbyte(dtype) and isinstance(dtype, DataType)

        if isinstance(expr, Var):
            return cast(self.uint8_pointer[expr], ~uint8), self.bit_offset[expr]
        elif isinstance(expr, Cast):
            original_type = self.type_infer(expr.expr)
            if is_addressable(original_type):
                if is_subbyte(get_base_type(original_type)):
                    return self.get_byte_and_bit_offset(expr.expr)
                else:
                    return cast(expr.expr, ~uint8), int32.zero
            else:
                raise NotImplementedError()
        elif isinstance(expr, (Add, Sub)):
            addressing = self.buf2addr[expr]
            dtype = addressing.type.as_data_type()
            uint8_ptr, bit_offset = self.get_byte_and_bit_offset(addressing.base)
            bit_index = bit_offset + int64(self.visit(addressing.offset)) * dtype.nbits
            new_uint8_ptr = uint8_ptr + bit_index // 8
            new_bit_offset = int32(bit_index % 8)
            return new_uint8_ptr, new_bit_offset
        elif isinstance(expr, Address):
            if isinstance(expr.expr, TensorElement):
                uint8_ptr, bit_offset = self.get_byte_and_bit_offset(expr.expr.base)
                assert len(expr.expr.indices) == 1
                merged_offset = bit_offset + int64(expr.expr.indices[0]) * dtype.nbits
                new_uint8_ptr = uint8_ptr + merged_offset // 8
                new_bit_offset = int32(merged_offset % 8)
                return new_uint8_ptr, new_bit_offset
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(expr)

    def update_var_definition(self, v: Var) -> Tuple[Var, Var]:
        if isinstance(v.type, TensorType) and is_subbyte(get_base_type(v.type)):
            ttype = v.type
            assert len(ttype.shape) == 1
            nbits = ttype.shape[0] * ttype.dtype.nbits
            new_shape = [(nbits + 7) // 8]
            new_ttype = TensorType(dtype=uint8, shape=new_shape, layout=row_major(*new_shape))
            new_var = var(v.hint, new_ttype)
            bit_offset_var = var(v.hint + "_bo", int32)
        elif isinstance(v.type, TensorPointerType) and is_subbyte(get_base_type(v.type)):
            ttype = v.type.tensor_type
            assert len(ttype.shape) == 1
            nbits = ttype.shape[0] * ttype.dtype.nbits
            new_shape = [(nbits + 7) // 8]
            new_tptype = TensorPointerType(TensorType(dtype=uint8, shape=new_shape, layout=row_major(*new_shape)))
            new_var = var(v.hint, new_tptype)
            bit_offset_var = var(v.hint + "_bo", int32)
        elif isinstance(v.type, PointerType) and is_subbyte(get_base_type(v.type)):
            new_type = ~uint8
            new_var = var(v.hint, new_type)
            bit_offset_var = var(v.hint + "_bo", int32)
        elif isinstance(v.type, DataType) and is_subbyte(v.type):
            raise NotImplementedError()
        else:
            assert False
        self.memo[v] = new_var
        self.uint8_pointer[v] = new_var
        self.bit_offset[v] = bit_offset_var
        return new_var, bit_offset_var

    def visit_DataType(self, t: DataType) -> DataType:
        if is_subbyte(t):
            return uint8
        else:
            return t

    def visit_PointerType(self, t: PointerType) -> BaseType:
        base_type = self.visit(t.base_type)
        if is_subbyte(base_type):
            return ~uint8
        return super().visit_PointerType(t)

    def visit_Function(self, func: Function) -> Function:
        self.analyzer.visit(func)
        for param in func.params:
            if is_addressable(param) and is_subbyte(get_base_type(param.type)):
                self.uint8_pointer[param] = cast(self.visit(param), ~uint8)
                self.bit_offset[param] = int32.zero
        return super().visit_Function(func)

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> Stmt:
        sb = StmtBuilder()

        if isinstance(stmt.var.type, TensorType) and is_subbyte(get_base_type(stmt.var.type)):
            new_var, ofs_var = self.update_var_definition(stmt.var)
            sb.declare(new_var, init=self.visit(stmt.init), scope=stmt.scope)
            sb.declare(ofs_var, init=int32.zero)
            return sb.finish()
        elif isinstance(stmt.var.type, TensorPointerType) and is_subbyte(get_base_type(stmt.var.type)):
            new_var, ofs_var = self.update_var_definition(stmt.var)
            if stmt.init is not None:
                uint8_pointer_init, bit_offset_init = self.get_byte_and_bit_offset(stmt.init)
            else:
                uint8_pointer_init, bit_offset_init = None, None
            sb.declare(new_var, init=uint8_pointer_init, scope=stmt.scope)
            sb.declare(ofs_var, init=bit_offset_init)
            return sb.finish()
        elif isinstance(stmt.var.type, PointerType) and is_subbyte(get_base_type(stmt.var.type)):
            sb = StmtBuilder()
            new_var, ofs_var = self.update_var_definition(stmt.var)
            if stmt.init is not None:
                uint8_pointer_init, bit_offset_init = self.get_byte_and_bit_offset(stmt.init)
            else:
                uint8_pointer_init, bit_offset_init = None, None
            sb.declare(new_var, init=uint8_pointer_init, scope=stmt.scope)
            sb.declare(ofs_var, init=bit_offset_init)
            return sb.finish()
        elif isinstance(stmt.var.type, DataType) and is_subbyte(stmt.var.type):
            raise NotImplementedError()
        else:
            return super().visit_DeclareStmt(stmt)

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        if isinstance(stmt.var.type, PointerType) and is_subbyte(get_base_type(stmt.var.type)):
            raise NotImplementedError()
        return super().visit_AssignStmt(stmt)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_var.type, TensorType) and is_subbyte(get_base_type(bind_var.type)):
                new_var, ofs_var = self.update_var_definition(bind_var)
                self.memo[bind_var] = new_var
                self.uint8_pointer[bind_var] = new_var
                self.bit_offset[bind_var] = int32.zero
                bind_vars.append(ofs_var)
                bind_values.append(int32.zero)
            elif isinstance(bind_var.type, TensorPointerType) and is_subbyte(get_base_type(bind_var.type)):
                raise NotImplementedError()
            elif isinstance(bind_var.type, PointerType) and is_subbyte(get_base_type(bind_var.type)):
                raise NotImplementedError()
            elif isinstance(bind_var.type, DataType) and is_subbyte(bind_var.type):
                raise NotImplementedError()
            bind_vars.append(self.visit(bind_var))
            bind_values.append(self.visit(bind_value))
        body = self.visit(stmt.body)
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(bind_vars, bind_values, body)

    def visit_TensorElement(self, e: TensorElement) -> Expr:
        if is_subbyte(self.type_infer(e)):
            uint8_pointer, bit_offset = self.get_byte_and_bit_offset(e.base)
            indices: List[Expr] = self.visit(e.indices)

            assert type_equal(self.type_infer(bit_offset), int32)
            assert len(indices) == 1
            sub_dtype: DataType = get_base_type(self.type_infer(e.base)).as_data_type()
            index = indices[0]

            return load_subbyte(uint8_pointer, bit_offset, index, sub_dtype.nbits)

        return super().visit_TensorElement(e)

    @staticmethod
    def update_uint8(uint8_ptr: Expr, idx: Union[Expr, int], uint8_mask: Expr, uint8_value: Expr) -> Stmt:
        sb = StmtBuilder()

        unchanged_mask = uint8(0xFF) ^ uint8_mask
        sb.buffer_store(uint8_ptr, indices=[idx], value=(uint8_value & uint8_mask) | (uint8_ptr[idx] & unchanged_mask))

        return sb.finish()

    def buffer_store_without_atomics(self, dtype: DataType, uint8_ptr: Expr, bit_offset: Expr, value: Expr) -> Stmt:
        sb = StmtBuilder()

        uint8_ptr = sb.declare(var("uint8_ptr", ~uint8), init=uint8_ptr)
        bit_offset = sb.declare(var("start_bit", int32), init=bit_offset)
        first_mask = sb.declare(
            var("first_mask", uint32), init=(((uint32.one << dtype.nbits) - uint32.one) << bit_offset) & uint8(0xFF)
        )

        # write the first byte
        sb += self.update_uint8(uint8_ptr, 0, first_mask, value << bit_offset)

        # write the second byte if the dtype value cross two bytes
        with sb.if_then(bit_offset > 8 - dtype.nbits):
            second_mask = sb.declare(
                var("second_mask", uint32), init=((uint32.one << (bit_offset - (8 - dtype.nbits))) - uint32.one)
            )
            sb += self.update_uint8(uint8_ptr, 1, second_mask, value >> (8 - bit_offset))

        return sb.finish()

    @staticmethod
    def atomic_update_uint16(uint16_ptr: Expr, idx: Union[Expr, int], uint16_mask: Expr, uint16_value: Expr) -> Stmt:
        """
        Update the bits in *uint16_ptr value indicated by the uint16_mask to the uint16_value, and keep the remaining
        bits unchanged, using atomic operations.
        """
        sb = StmtBuilder()

        uint16_value = sb.declare_var("value", uint16, uint16_value & uint16_mask)
        unchanged_mask = sb.declare_var("unchanged_mask", uint16, init=uint16(0xFFFF) ^ uint16_mask)

        original = sb.declare(var("original", uint16))
        result = sb.declare(var("result", uint16))
        with sb.while_loop(boolean.true):
            sb.assign(original, value=uint16_ptr[idx])
            updated_value = uint16_value | (original & unchanged_mask)
            updated = sb.declare(var("updated", uint16), init=updated_value)
            sb.assign(result, value=atomic_cas(uint16_ptr + idx, compare=original, value=updated))
            with sb.if_then(result == original):
                sb.brk()

        return sb.finish()

    @staticmethod
    def atomic_update_uint32(uint32_ptr: Expr, idx: Union[Expr, int], uint32_mask: Expr, uint32_value: Expr) -> Stmt:
        """
        Update the bits in *uint32_ptr value indicated by the uint32_mask to the uint32_value, and keep the remaining
        bits unchanged, using atomic operations.
        """
        sb = StmtBuilder()

        uint32_value = sb.declare_var("value", uint32, uint32_value & uint32_mask)
        unchanged_mask = sb.declare_var("unchanged_mask", uint32, init=uint32(0xFFFFFFFF) ^ uint32_mask)

        original = sb.declare(var("original", uint32))
        result = sb.declare(var("result", uint32))
        with sb.while_loop(boolean.true):
            sb.assign(original, value=uint32_ptr[idx])
            updated_value = uint32_value | (original & unchanged_mask)
            updated = sb.declare(var("updated", uint32), init=updated_value)
            sb.assign(result, value=atomic_cas(uint32_ptr + idx, compare=original, value=updated))
            with sb.if_then(result == original):
                sb.brk()

        return sb.finish()

    def buffer_store_with_atomics(self, dtype: DataType, uint8_ptr: Expr, bit_offset: Expr, value: Expr) -> Stmt:
        if get_current_target().is_nvgpu():
            # nvidia supports 16-bit atomic cas, but I did not test the performance of using 32-bit and 16-bit atomic cas
            # if there is not much difference in performance, switch to 32-bit atomic cas for consistency with amd gpu
            sb = StmtBuilder()

            addr = sb.declare_var("addr", int64, init=cast(uint8_ptr, int64))
            uint16_ptr = sb.declare_var("uint16_ptr", ~uint16, init=cast((addr // 2 * 2), ~uint16))
            bit_offset = sb.declare_var("bit_offset", int32, init=bit_offset + (addr % 2) * 8)  # bit offset in uint16
            first_mask = sb.declare_var(
                "first_mask", uint16, init=(((uint32.one << dtype.nbits) - uint32.one) << bit_offset) & uint16(0xFFFF)
            )
            first_value = sb.declare_var("first_value", uint16, init=(value << bit_offset) & first_mask)

            sb += self.atomic_update_uint16(uint16_ptr, 0, first_mask, first_value)

            with sb.if_then(bit_offset > 16 - dtype.nbits):
                second_mask = sb.declare_var(
                    "second_mask", uint16, init=((uint32.one << (dtype.nbits - (16 - bit_offset))) - uint32.one)
                )
                second_value = sb.declare_var("second_value", uint16, init=(value >> (16 - bit_offset)) & second_mask)
                sb += self.atomic_update_uint16(uint16_ptr, 1, second_mask, second_value)
            return sb.finish()
        elif get_current_target().is_amdgpu():
            # amd gpu only supports 32-bit(+) atomic cas from the hip documentation
            sb = StmtBuilder()

            addr = sb.declare_var("addr", int64, init=cast(uint8_ptr, int64))
            uint32_ptr = sb.declare_var("uint32_ptr", ~uint32, init=cast((addr // 4 * 4), ~uint32))
            bit_offset = sb.declare_var("bit_offset", int32, init=bit_offset + (addr % 4) * 8)  # bit offset in uint32
            first_mask = sb.declare_var(
                "first_mask",
                uint32,
                init=(((uint32.one << dtype.nbits) - uint32.one) << bit_offset) & uint32(0xFFFFFFFF),
            )
            first_value = sb.declare_var("first_value", uint32, init=(value << bit_offset) & first_mask)

            sb += self.atomic_update_uint32(uint32_ptr, 0, first_mask, first_value)

            with sb.if_then(bit_offset > 32 - dtype.nbits):
                second_mask = sb.declare_var(
                    "second_mask", uint32, init=((uint32.one << (dtype.nbits - (32 - bit_offset))) - uint32.one)
                )
                second_value = sb.declare_var("second_value", uint32, init=(value >> (32 - bit_offset)) & second_mask)
                sb += self.atomic_update_uint32(uint32_ptr, 1, second_mask, second_value)
            return sb.finish()
        else:
            raise NotImplementedError()

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt) -> Stmt:
        if is_subbyte(get_base_type(self.type_infer(stmt.buf))):
            uint8_pointer, bit_offset = self.get_byte_and_bit_offset(stmt.buf)
            indices: List[Expr] = self.visit(stmt.indices)
            value: Expr = self.visit(stmt.value)

            assert type_equal(self.type_infer(bit_offset), int32)
            assert self.type_infer(value) == uint8
            assert len(indices) == 1

            sub_dtype: DataType = get_base_type(self.type_infer(stmt.buf)).as_data_type()
            index: Expr = indices[0]

            updated_uint8_pointer = uint8_pointer + int64(index) * sub_dtype.nbits // 8
            updated_bit_offset = int32((bit_offset + int64(index) * sub_dtype.nbits) % 8)

            sb = StmtBuilder()

            value = sb.declare_var("value", uint8, init=value)

            if self.buf2addr[stmt.buf].scope in [DeclareScope.Register]:
                sb.append(
                    self.buffer_store_without_atomics(sub_dtype, updated_uint8_pointer, updated_bit_offset, value)
                )
            elif self.buf2addr[stmt.buf].scope in [DeclareScope.Global, DeclareScope.Shared]:
                sb.append(self.buffer_store_with_atomics(sub_dtype, updated_uint8_pointer, updated_bit_offset, value))
            else:
                raise NotImplementedError(str(self.buf2addr[stmt.buf].scope) + " " + str(stmt.buf))

            return sb.finish()

        return super().visit_BufferStoreStmt(stmt)

    def visit_Constant(self, e: Constant) -> Expr:
        if is_subbyte(self.type_infer(e)):
            dtype = self.type_infer(e)
            if dtype.is_float_subbyte():
                return cast_subbyte_float_from_f32(float32(e.value), dtype)
            else:
                return self._cast_int32_to_subbyte_integer(int32(e.value), dtype)
        return super().visit_Constant(e)

    def _cast_subbyte_integer_to_int32(self, e: Expr, src_dtype: DataType) -> Expr:
        assert self.type_infer(e) == uint8
        assert isinstance(src_dtype, IntegerSubbyteType), src_dtype
        nbits: int = src_dtype.nbits
        e = cast(e, int32)
        if src_dtype.signedness():
            e = (e << (32 - nbits)) >> (32 - nbits)
        return e

    def _cast_int32_to_subbyte_integer(self, e: Expr, dst_dtype: DataType) -> Expr:
        assert self.type_infer(e) == int32 and dst_dtype.is_integer_subbyte()
        assert isinstance(dst_dtype, IntegerSubbyteType), dst_dtype
        nbits: int = dst_dtype.nbits
        return cast(e, uint8) & uint8((1 << nbits) - 1)

    def visit_Cast(self, e: Cast) -> Expr:
        src_type = self.type_infer(e.expr)
        dst_type = self.type_infer(e)

        if is_subbyte(src_type) or is_subbyte(dst_type):
            src_dtype: DataType = src_type
            dst_dtype: DataType = dst_type
            assert isinstance(src_dtype, DataType) and isinstance(dst_dtype, DataType)
            if src_dtype.is_float_subbyte() and dst_type == float32:
                return cast_subbyte_float_to_f32(self.visit(e.expr), src_dtype)
            elif src_dtype == float32 and dst_dtype.is_float_subbyte():
                return cast_subbyte_float_from_f32(self.visit(e.expr), dst_dtype)
            elif src_dtype.is_integer_subbyte() and dst_type == int32:
                return self._cast_subbyte_integer_to_int32(self.visit(e.expr), src_dtype)
            elif src_dtype == int32 and dst_dtype.is_integer_subbyte():
                return self._cast_int32_to_subbyte_integer(self.visit(e.expr), dst_dtype)
            else:
                # try using a proxy data type
                proxy_dtype: DataType
                if src_dtype.is_subbyte():
                    if src_dtype.is_float():
                        proxy_dtype = float32
                    elif src_dtype.is_integer():
                        proxy_dtype = int32
                    else:
                        assert False
                elif dst_dtype.is_subbyte():
                    if dst_dtype.is_float():
                        proxy_dtype = float32
                    elif dst_dtype.is_integer():
                        proxy_dtype = int32
                    else:
                        assert False
                else:
                    assert False
                if src_dtype == proxy_dtype or dst_dtype == proxy_dtype:
                    # we can not use float32 or int32 as a proxy data type
                    raise NotImplementedError("Please implement the cast from {} to {}".format(src_dtype, dst_dtype))
                return self.visit(cast(cast(e.expr, proxy_dtype), dst_dtype))
        elif (
            is_addressable(src_type)
            and is_subbyte(get_base_type(src_type))
            and is_addressable(dst_type)
            and not is_subbyte(get_base_type(dst_type))
        ):
            # cast a sub-byte type pointer to normal pointer
            sb = StmtBuilder()
            uint8_ptr, bit_offset = self.get_byte_and_bit_offset(e.expr)
            uint8_ptr = sb.declare_var("uint8_ptr", ~uint8, init=uint8_ptr)
            bit_offset = sb.declare_var("bit_offset", int32, init=bit_offset)
            sb.assertion(
                cond=bit_offset == 0, msg="Casting {} to {}, bit offset must be zero".format(src_type, dst_type)
            )
            ret = sb.declare_var("casted_type", dst_type, init=cast(uint8_ptr, dst_type))
            self.append_prologue_stmt(sb.finish())
            return ret

        return super().visit_Cast(e)


class LowerSubbyteTypePass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        # AddressingAnalyzer().debug_analyze_and_print(func)
        rewriter = LowerSubbyteTypeRewriter()
        return rewriter.visit(func)


def lower_subbyte_type_pass() -> LowerSubbyteTypePass:
    return LowerSubbyteTypePass()
