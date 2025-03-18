from __future__ import annotations

from typing import Optional, Union, Type, Any, Tuple, Sequence
import types
import operator
import math
import ast
import inspect
import builtins

from hidet import ir as hidet_ir
from hidet.ir.analyzers import normalize_launch_dims
from hidet.ir.type import BaseType, data_type
from hidet.ir.expr import Var
from hidet.lang.script import eliminate_indent, eliminate_decorators
from hidet.lang.transpiler import PythonAstFunctor, HidetProgramError
from tilus import ir as tilus_ir
from tilus.extensions.hidet.ir.expr import convert_to_expr
from tilus.ir.layout import Layout
from tilus.ir.func import Function
from tilus.ir.stmt import Stmt, SeqStmt, InstructionStmt
from tilus.ir.inst import Instruction, AllocateScalarInst
from tilus.ir.builders import IRBuilder
from tilus.ir.value import Value
from tilus.lang.script import Script


class TilusProgramError(HidetProgramError):
    pass


class Scope:
    def __init__(self, parent: Optional[Scope]):
        self.parent: Optional[Scope] = parent
        self.name2var: dict[str, Var] = {}
        self.name2value: dict[str, Value] = {}
        self.name2host_var: dict[str, Any] = {}
        self.stmts: list[Stmt] = []
        self.attributes: dict[str, Any] = {}

    @staticmethod
    def default_top_level():
        scope = Scope(None)
        return scope

    def bind(self, name: str, var_or_value: Var | Value | Any):
        if isinstance(var_or_value, Var):
            self.name2var[name] = var_or_value
        elif isinstance(var_or_value, Value):
            self.name2value[name] = var_or_value
        else:
            self.name2host_var[name] = var_or_value
        # print('binding {} with {}'.format(name, var_or_value))

    def lookup(self, name: str, search_parents: bool = True) -> Var | Value | Any | None:
        if name in self.name2var:
            return self.name2var[name]
        if name in self.name2value:
            return self.name2value[name]
        if name in self.name2host_var:
            return self.name2host_var[name]
        if search_parents and self.parent:
            return self.parent.lookup(name, search_parents)
        return None

    def annotate(self, name: str, value: Any):
        if name in self.attributes:
            raise ValueError("Attribute {} has already been annotated.".format(name))
        self.attributes[name] = value

    def append(self, inst_or_stmt: Instruction | Stmt):
        stmt = inst_or_stmt if isinstance(inst_or_stmt, Stmt) else InstructionStmt(inst_or_stmt)
        self.stmts.append(stmt)

    def flush_stmts(self) -> Stmt:
        seq_stmt = SeqStmt.create(seq=self.stmts)
        self.stmts.clear()
        return seq_stmt


class ScopeStack:
    def __init__(self) -> None:
        self.scopes: list[Scope] = [Scope.default_top_level()]

    def __enter__(self) -> Scope:
        parent = self.scopes[-1]
        scope = Scope(parent)
        self.scopes.append(scope)
        return scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class LambdaProxy:
    def __init__(self, lambda_expr: ast.Lambda, translator):
        self.lambda_expr: ast.Lambda = lambda_expr
        self.translator: Transpiler = translator

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise HidetProgramError(
                self.translator, self.lambda_expr, "Do not support keyword arguments in lambda function."
            )

        with self.translator.scope() as lambda_params_scope:
            if len(args) != len(self.lambda_expr.args.args):
                raise HidetProgramError(
                    self.translator,
                    self.lambda_expr,
                    "The number of arguments does not match the lambda function definition.",
                )
            for arg, arg_expr in zip(self.lambda_expr.args.args, args):
                arg_name = arg.arg
                lambda_params_scope.bind(arg_name, arg_expr)
            return self.translator.visit(self.lambda_expr.body)


class Transpiler(PythonAstFunctor):
    def __init__(self) -> None:
        super().__init__(file="", start_lineno=0, start_column=0)
        self.ib = IRBuilder()
        self.scope_stack = ScopeStack()
        self.method_annotations: dict[str, Any] = {}

        self._script: Optional[Script] = None

    def scope(self) -> ScopeStack:
        return self.scope_stack

    @property
    def current_scope(self) -> Scope:
        return self.scope_stack.scopes[-1]

    @property
    def script(self) -> Script:
        if self._script is None:
            raise RuntimeError("The script is not set.")
        return self._script

    def transpile(self, script: Script, method: types.FunctionType) -> Function:
        # Extract the source code of given function
        lines, start_line = inspect.getsourcelines(method)
        file: Optional[str] = inspect.getsourcefile(method)
        if file is None:
            raise RuntimeError('Can not get the source file of the given function "{}".'.format(method.__name__))

        source = "".join(lines)
        source, col_offset = eliminate_indent(source)
        source, inc_lineno = eliminate_decorators(source)
        start_line += inc_lineno
        parsed: ast.AST = ast.parse(source=source)
        self.file = file
        self.start_lineno = start_line
        self.start_column = col_offset

        # Get the environment (globals and binding of free variables)
        # See the data model of python for the details of func.__globals__, func.__closure__ and func.__code__:
        #     https://docs.python.org/3/reference/datamodel.html
        env: dict[str, Any] = method.__globals__.copy()
        func_freevar_names: list[str] = list(method.__code__.co_freevars)
        func_freevar_cells: list[Any] = [v.cell_contents for v in method.__closure__] if method.__closure__ else []
        assert len(func_freevar_names) == len(func_freevar_cells)
        env.update(dict(zip(func_freevar_names, func_freevar_cells)))

        # get the type annotations of function parameters.
        self.method_annotations = dict(method.__annotations__.items())
        with self.scope() as env_scope:
            for name, value in env.items():
                env_scope.bind(name, value)
            env_scope.bind("self", script)

            script._transpiler = self
            self._script = script

            function = self.visit(parsed)
            assert isinstance(function, Function)

            # prevent loop reference
            self._script = None
            script._transpiler = None

            return function

    def process_assign(
        self, lhs: Union[ast.Attribute, ast.Subscript, ast.Name], rhs, type_annotation: Optional[ast.expr] = None
    ):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        # check the rhs value, must be an instance of rhs_allowed_types or a list of these kinds of elements.
        host_var_types: Tuple[Any, ...] = (Layout, str, list, tuple, dict)
        var_types = (hidet_ir.Expr, tilus_ir.Value, float, int, str, type(None))
        rhs_allowed_types = var_types + host_var_types
        assert isinstance(rhs, rhs_allowed_types), 'unexpected value "{}" with type {}'.format(rhs, type(rhs))

        # three cases of assignment:
        #    1. v = ...
        #    2. a[i, j] = ...
        #    3. attr.name = ...
        if isinstance(lhs, ast.Name):
            var_name: str = lhs.id
            lookup_result = self.current_scope.lookup(var_name, search_parents=True)
            if lookup_result is None:
                # bind a new name to the right side, the rhs could be
                #  1) a hidet expression => we define a new scalar variable
                #  2) a tilus value => we bind the value to the name
                #  3) other host expressions
                #    3.1) if there is type annotation, we define a scalar variable
                #    3.2) otherwise, we bind the host expression to the name
                if isinstance(lhs, hidet_ir.Expr):
                    declare_inst = AllocateScalarInst.create(hint=var_name, scalar_type=hidet_ir.infer_type(rhs))
                    self.current_scope.append(declare_inst)
                    self.current_scope.bind(var_name, rhs)
                elif isinstance(rhs, tilus_ir.Value):
                    self.current_scope.bind(var_name, rhs)
                else:
                    if type_annotation is not None:
                        raise NotImplementedError("define var with type annotation")
                    else:
                        if rhs is None:
                            raise TilusProgramError(
                                self, lhs, "Trying to assign None to a variable, which is not allowed."
                            )
                        self.current_scope.bind(var_name, rhs)
            else:
                raise NotImplementedError()
        elif isinstance(lhs, ast.Subscript):
            # example: a[3, 4] = 5.0
            raise NotImplementedError("subscript assignment")
        elif isinstance(lhs, ast.Attribute):
            # example: self.attrs.blocks = 16, 16
            lhs_base = self.visit(lhs.value)

            namespace = {
                self.script.attrs: "",
            }
            if lhs_base in namespace:
                attr_name = namespace[lhs_base] + lhs.attr
                if attr_name in [
                    "warps",
                    "blocks",
                ]:
                    if isinstance(rhs, (tuple, list)):
                        rhs = [hidet_ir.tools.simplify(v) for v in rhs]
                    else:
                        rhs = hidet_ir.tools.simplify(rhs)
                self.current_scope.annotate(attr_name, rhs)
            else:
                raise HidetProgramError(self, lhs, "Invalid assignment.")
        else:
            type_name = type(lhs).__name__
            raise HidetProgramError(self, lhs, 'Cannot recognize "{}" as left side of assignment.'.format(type_name))

    def visit_Module(self, module: ast.Module):
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            msg = "The module expects to have only one function definition statement, got\n"
            msg += str(ast.unparse(module))
            raise ValueError(msg)
        return self.visit(module.body[0])

    def process_param_ret_type(self, arg, arg_type: Union[BaseType, Type[int], Type[float], Type[bool]]):
        if isinstance(arg_type, BaseType):
            return arg_type
        elif arg_type in [bool, int, float]:
            type_dict = {bool: data_type("bool"), int: data_type("int32"), float: data_type("float32")}
            arg_type = type_dict[arg_type]
        elif isinstance(arg_type, str):
            raise TilusProgramError(
                self,
                arg,
                (
                    "A python string as parameter type annotation detected. \n"
                    'This is usually because "from __future__ import annotations" has been used.\n'
                    "Currently, tilus script is not compatible with this feature. \n"
                    "Please considering not using it in module that defines tilus script."
                ),
            )
        else:
            raise TilusProgramError(self, arg, "Tilus expect a type annotation for this parameter.")
        return arg_type

    def visit_FunctionDef(self, func_def: ast.FunctionDef):
        func_params = []
        with self.scope() as scope:
            # process function arguments
            args: ast.arguments = func_def.args

            # make sure that the function parameters only have normal positional arguments
            if args.vararg is not None:
                raise TilusProgramError(self, args.vararg, 'Tilus program does not support "*args" arguments.')
            if len(args.kwonlyargs) != 0:
                raise TilusProgramError(self, args.kwonlyargs[0], 'Tilus program does not support "*kwargs" arguments.')
            if args.kwarg is not None:
                raise TilusProgramError(self, args.kwarg, "Tilus program does not support keyword arguments.")
            if len(args.kw_defaults) > 0:
                raise TilusProgramError(self, args.kw_defaults[0], "Tilus does not support default argument.")
            if len(args.defaults) > 0:
                raise TilusProgramError(self, args.defaults[0], "Tilus does not support default argument.")

            for idx, arg in enumerate(args.args):
                arg_name = arg.arg

                if idx == 0 and arg_name == "self":
                    continue
                if arg_name not in self.method_annotations:
                    raise TilusProgramError(self, arg, "Tilus expects type annotation for each function parameter.")

                arg_type = self.method_annotations[arg_name]
                processed_arg_type: BaseType = self.process_param_ret_type(arg, arg_type)
                param_var = Var(hint=arg_name, type=processed_arg_type)
                func_params.append(param_var)
                scope.bind(arg_name, param_var)

            # return type
            if func_def.returns is not None:
                raise TilusProgramError(self, func_def.returns, "Tilus does not support return type annotation.")

            # process function body
            for stmt in func_def.body:
                self.visit(stmt)

            # process the attributes
            attributes = scope.attributes
            if "blocks" not in attributes:
                msg = """
                Tilus script should set the number of blocks via self.blocks = ... like
                    self.blocks = dim_x
                or 
                    self.blocks = dim_x, dim_y
                """
                raise TilusProgramError(self, func_def, msg)
            blocks = [convert_to_expr(dim) for dim in normalize_launch_dims(self.current_scope.attributes["blocks"])]
            if "warps" not in attributes:
                raise TilusProgramError(
                    self, func_def, "Tilus script should set the number of warps via self.warps = ..."
                )
            warps = self.current_scope.attributes["warps"]

            return Function.create(
                name=func_def.name,
                params=func_params,
                num_warps=warps,
                num_blocks=blocks,
                body=scope.flush_stmts(),
                annotations={},
            )

    def visit_Expr(self, expr: ast.Expr):
        value = self.visit(expr.value)

        if value is None:
            return
        else:
            raise NotImplementedError(value)

    def visit_Call(self, expr: ast.Call):
        func = self.visit(expr.func)
        args = []
        for arg in expr.args:
            if isinstance(arg, ast.Starred):
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
        if len(expr.keywords) == 0:
            kwargs = {}
        elif len(expr.keywords) == 1 and expr.keywords[0].arg is None:
            # func(a, b, **kwargs)
            kwargs = self.visit(expr.keywords[0].value)
        else:
            # func(a=1, b=2, c=3)
            kwargs = {kwarg.arg: self.visit(kwarg.value) for kwarg in expr.keywords}

        if isinstance(func, types.FunctionType):
            # call python function
            return func(*args, **kwargs)
        elif isinstance(func, types.MethodType):
            # call python class method
            return func(*args, **kwargs)
        elif isinstance(func, (types.BuiltinMethodType, types.BuiltinFunctionType)):
            # call python builtin method, such "a string".format(...) or max, min
            from hidet import ir
            from hidet.ir import primitives

            if all(not isinstance(arg, ir.Node) for arg in args):
                # pure python function call
                return func(*args, **kwargs)
            else:
                if any(not isinstance(arg, (ir.Expr, int, float, bool)) for arg in args):
                    # if any argument is not a valid expression
                    return func(*args, **kwargs)
                # overload hidet primitive, such as max, min
                func_map = {
                    builtins.max: (2, primitives.max),
                    builtins.min: (2, primitives.min),
                    math.exp: (1, primitives.exp),
                    math.log: (1, primitives.log),
                    math.sqrt: (1, primitives.sqrt),
                    math.sin: (1, primitives.sin),
                    math.cos: (1, primitives.cos),
                    math.tan: (1, primitives.tan),
                    math.asin: (1, primitives.asin),
                    math.acos: (1, primitives.acos),
                    math.atan: (1, primitives.atan),
                    math.sinh: (1, primitives.sinh),
                    math.cosh: (1, primitives.cosh),
                    math.tanh: (1, primitives.tanh),
                    math.asinh: (1, primitives.asinh),
                    math.acosh: (1, primitives.acosh),
                    math.atanh: (1, primitives.atanh),
                    math.ceil: (1, primitives.ceil),
                    math.floor: (1, primitives.floor),
                    math.trunc: (1, primitives.trunc),
                    math.isnan: (1, primitives.isnan),
                    math.isinf: (1, primitives.isinf),
                }
                if len(kwargs) > 0:
                    msg = "Hidet do not support calling builtin function with keyword argument."
                    raise HidetProgramError(self, expr, msg)
                if func in func_map:
                    arity, hidet_func = func_map[func]  # type: ignore[index]
                    if len(args) != arity:
                        msg = f'Hidet builtin function "{func.__name__}" takes {arity} arguments.'
                        raise HidetProgramError(self, expr, msg)
                    return hidet_func(*args)  # type: ignore[operator]
                else:
                    raise HidetProgramError(
                        self,
                        expr,
                        'Currently, do not support calling python builtin function "{}".'.format(func.__qualname__),
                    )
        else:
            return func(*args, **kwargs)

    def visit_Attribute(self, expr: ast.Attribute):
        base = self.visit(expr.value)
        attr = expr.attr

        self_attributes = {(self._script, "blockIdx")}
        if (base, attr) in self_attributes:
            # self.blockIdx
            from hidet.ir.primitives.cuda.vars import blockIdx

            self_attributes_map = {(self._script, "blockIdx"): blockIdx}
            ret = self_attributes_map[(base, attr)]
        elif hasattr(base, attr):
            ret = getattr(base, attr)
        else:
            raise HidetProgramError(self, expr, 'Can not access attribute "{}" of this object.'.format(attr))
        # print('accessing attribute {} of {}: {}'.format(attr, base, ret))
        return ret

    def visit_Name(self, expr: ast.Name):
        if isinstance(expr.ctx, ast.Store):
            raise ValueError("Internal Error, please deal with all Store behavior in parent nodes like Assign.")
        elif isinstance(expr.ctx, ast.Load):
            name: str = expr.id
            var = self.current_scope.lookup(name)
            if var is None:
                if name in builtins.__dict__:
                    # access builtin functions such as max, min
                    return getattr(builtins, name)
                raise HidetProgramError(self, expr, "Trying to access variable without definition.")
            return var
        elif isinstance(expr.ctx, ast.Del):
            raise HidetProgramError(self, expr, "Hidet does not support del statement.")
        else:
            raise ValueError()

    def visit_Tuple(self, expr: ast.Tuple):
        return (self.visit(v) for v in expr.elts)

    def visit_List(self, expr: ast.List):
        return [self.visit(v) for v in expr.elts]

    def visit_BinOp(self, expr: ast.BinOp):
        from hidet import ir

        lhs = self.visit(expr.left)
        rhs = self.visit(expr.right)
        if isinstance(lhs, str) and isinstance(rhs, str):
            assert isinstance(expr.op, ast.Add)
            return lhs + rhs
        elif isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
            assert isinstance(expr.op, ast.Add)
            return list(lhs) + list(rhs)
        elif isinstance(lhs, (ir.Expr, float, int)) and isinstance(rhs, (ir.Expr, float, int)):
            from hidet.ir import primitives

            op_dict = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.BitXor: operator.xor,
                ast.BitOr: operator.or_,
                ast.BitAnd: operator.and_,
                ast.Pow: primitives.pow,
                ast.LShift: operator.lshift,
                ast.RShift: operator.rshift,
            }

            if type(expr.op) in op_dict:
                return op_dict[type(expr.op)](lhs, rhs)
            else:
                type_name = type(expr.op).__name__
                raise HidetProgramError(self, expr, "Currently, we do not support {} operator.".format(type_name))
        else:
            raise HidetProgramError(
                self, expr, "Can not apply operator {} to {} and {}.".format(expr.op, type(lhs), type(rhs))
            )

    def visit_Assign(self, stmt: ast.Assign):
        if len(stmt.targets) > 1:
            raise HidetProgramError(self, stmt, 'Hidet does not support syntax like "a = b = 1".')
        target = stmt.targets[0]
        value = stmt.value

        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            # a, b = c, d
            lhs_list = target.elts
            rhs_list = [self.visit(v) for v in value.elts]
            if len(lhs_list) != len(rhs_list):
                raise HidetProgramError(self, stmt, "The number of left values and right values does not match.")
            for lhs, rhs in zip(lhs_list, rhs_list):
                assert isinstance(lhs, (ast.Attribute, ast.Subscript, ast.Name))
                self.process_assign(lhs, rhs)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # a, b = c
            lhs_list = target.elts
            rhs_list = self.visit(value)
            if len(lhs_list) != len(rhs_list):
                raise HidetProgramError(self, stmt, "The number of left values and right values does not match.")
            for lhs, rhs in zip(lhs_list, rhs_list):
                assert isinstance(lhs, (ast.Attribute, ast.Subscript, ast.Name))
                self.process_assign(lhs, rhs)
        elif isinstance(value, (ast.Tuple, ast.List)):
            # a = c, d
            rhs_list = [self.visit(v) for v in value.elts]
            assert isinstance(target, (ast.Attribute, ast.Subscript, ast.Name))
            self.process_assign(target, rhs_list)
        else:
            # a = c
            assert isinstance(target, (ast.Attribute, ast.Subscript, ast.Name))
            rhs = self.visit(value)
            self.process_assign(target, rhs)

    def visit_Lambda(self, expr: ast.Lambda):
        return LambdaProxy(expr, self)

    def visit_Subscript(self, expr: ast.Subscript):
        base = self.visit(expr.value)
        indices = self.visit(expr.slice)

        if isinstance(base, Sequence):
            return base[indices]
        else:
            raise NotImplementedError()

    def visit_Constant(self, expr: ast.Constant):
        if isinstance(expr.value, (float, int)):
            return expr.value
        elif isinstance(expr.value, str):
            return expr.value
        elif expr.value is None:
            return expr.value
        else:
            raise HidetProgramError(self, expr, "Can not recognize Constant {}".format(repr(expr.value)))

    def visit_Compare(self, expr: ast.Compare):
        front = self.visit(expr.left)
        op_dict = {
            ast.And: hidet_ir.logical_and,
            ast.Or: hidet_ir.logical_or,
            ast.Eq: hidet_ir.equal,
            ast.Gt: lambda a, b: hidet_ir.less_than(b, a),  # pylint: disable=arguments-out-of-order
            ast.Lt: hidet_ir.less_than,
            ast.GtE: lambda a, b: hidet_ir.less_equal(b, a),  # pylint: disable=arguments-out-of-order
            ast.LtE: hidet_ir.less_equal,
            ast.NotEq: hidet_ir.not_equal,
        }
        py_op_dict = {
            ast.And: operator.and_,
            ast.Or: operator.or_,
            ast.Eq: operator.eq,
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.GtE: operator.ge,
            ast.LtE: operator.le,
            ast.NotEq: operator.ne,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }
        cond = None
        comparators = [self.visit(v) for v in expr.comparators]
        for op, current in zip(expr.ops, comparators):
            op_kind = type(op)
            if isinstance(front, hidet_ir.Node) or isinstance(current, hidet_ir.Node):
                if op_kind not in op_dict:
                    raise HidetProgramError(
                        self, expr, "Currently, we do not support {} operator for hidet vars.".format(op_kind.__name__)
                    )
                cur_cond = op_dict[op_kind](front, current)  # type: ignore[operator]
            else:
                cur_cond = py_op_dict[op_kind](front, current)
            cond = hidet_ir.logical_and(cond, cur_cond) if cond is not None else cur_cond
            front = current
        return cond
