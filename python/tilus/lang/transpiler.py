from __future__ import annotations

import ast
import builtins
import inspect
import math
import operator
import types
from typing import Any, Optional, Sequence, Tuple, Type, Union

import tilus.lang.constructs.loops
from hidet import ir as hidet_ir
from hidet.ir.analyzers import normalize_launch_dims
from hidet.ir.expr import Constant, Var
from hidet.ir.type import BaseType, data_type
from hidet.lang.script import eliminate_decorators, eliminate_indent
from hidet.lang.transpiler import HidetProgramError, PythonAstFunctor
from tilus import ir as tilus_ir
from tilus.extensions.hidet.ir.expr import as_expr
from tilus.ir import RegisterTensor
from tilus.ir.builders import IRBuilder, StmtBuilder
from tilus.ir.func import Function, Metadata
from tilus.ir.inst import Instruction
from tilus.ir.instructions import AssignInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.stmt import AssignStmt, DeclareStmt, EvaluateStmt, InstStmt, SeqStmt, Stmt
from tilus.ir.tensor import Tensor
from tilus.lang.constructs.loops import TilusLoopIterable
from tilus.lang.script import Script


class TilusProgramError(HidetProgramError):
    pass


class Scope:
    def __init__(self, parent: Optional[Scope]):
        self.parent: Optional[Scope] = parent
        self.name2var: dict[str, Var] = {}
        self.name2value: dict[str, Tensor] = {}
        self.name2host_var: dict[str, Any] = {}
        self.stmts: list[Stmt] = []
        self.attributes: dict[str, Any] = {}

    @staticmethod
    def default_top_level():
        scope = Scope(None)
        # when user use range(...), it will be translated to tilus.lang.constructs.loops.range(...)
        scope.bind("range", tilus.lang.constructs.loops.range)
        return scope

    def bind(self, name: str, var_or_value: Var | Tensor | Any) -> None:
        if isinstance(var_or_value, Var):
            self.name2var[name] = var_or_value
        elif isinstance(var_or_value, Tensor):
            self.name2value[name] = var_or_value
        else:
            self.name2host_var[name] = var_or_value
        # print('binding {} with {}'.format(name, var_or_value))

    def lookup(self, name: str, search_parents: bool = True) -> Var | Tensor | Any | None:
        if name in self.name2var:
            return self.name2var[name]
        if name in self.name2value:
            return self.name2value[name]
        if name in self.name2host_var:
            return self.name2host_var[name]
        if search_parents and self.parent:
            return self.parent.lookup(name, search_parents)
        return None

    def annotate(self, name: str, value: Any) -> None:
        if name in self.attributes:
            raise ValueError("Attribute {} has already been annotated.".format(name))
        self.attributes[name] = value

    def append(self, inst_or_stmt: Instruction | Stmt) -> None:
        stmt = inst_or_stmt if isinstance(inst_or_stmt, Stmt) else InstStmt(inst_or_stmt)
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
    def __init__(self, lambda_expr: ast.Lambda, translator: Transpiler):
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
        self.type_annotations: dict[str, Any] = {}
        self.name2consts: dict[str, Constant] = {}
        self.name2divisibility: dict[str, int] = {}

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

    def transpile(
        self, script: Script, name2consts: dict[str, Constant], name2divisibility: dict[str, int]
    ) -> Function:
        # Extract the source code of given function
        method = script.__class__.__call__
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
        self.type_annotations = dict(method.__annotations__.items())
        self.name2consts = name2consts
        self.name2divisibility = name2divisibility
        with self.scope() as env_scope:
            for name, value in env.items():
                env_scope.bind(name, value)
            env_scope.bind("self", script)

            script._builder = StmtBuilder()
            self._script = script

            function = self.visit(parsed)
            assert isinstance(function, Function)

            # prevent loop reference
            self._script = None
            script._builder = None

            return function

    def process_assign(
        self, lhs: Union[ast.Attribute, ast.Subscript, ast.Name], rhs: Any, type_annotation: Optional[ast.expr] = None
    ) -> None:
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        # check the rhs value, must be an instance of rhs_allowed_types or a list of these kinds of elements.
        host_var_types: Tuple[Any, ...] = (RegisterLayout, str, list, tuple, dict)
        var_types = (hidet_ir.Expr, tilus_ir.Tensor, float, int, str, type(None))
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
                if isinstance(rhs, hidet_ir.Expr):
                    stmt = DeclareStmt(var=Var(hint=var_name, type=hidet_ir.infer_type(rhs)), init=rhs)
                    self.current_scope.append(stmt)
                    self.current_scope.bind(var_name, stmt.var)
                elif isinstance(rhs, tilus_ir.Tensor):
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
                # assignment
                if isinstance(lookup_result, Var):
                    if not isinstance(rhs, (hidet_ir.Expr, int, float, str)):
                        raise TilusProgramError(self, lhs, "Assignment between Var is only accepted for hidet_ir.Expr.")
                    self.current_scope.append(AssignStmt(var=lookup_result, value=as_expr(rhs)))
                elif isinstance(lookup_result, Tensor):
                    if not isinstance(rhs, RegisterTensor) or not isinstance(lookup_result, RegisterTensor):
                        raise TilusProgramError(
                            self, lhs, "Assignment between Value is only accepted for RegisterValue."
                        )
                    from tilus.extensions.hidet.ir.type import type_equal

                    if not type_equal(lookup_result.dtype, rhs.dtype):
                        raise TilusProgramError(
                            self,
                            lhs,
                            "Different types of RegisterValue are not allowed to be assigned to each other. ",
                        )
                    self.current_scope.append(AssignInst.create(output=lookup_result, x=rhs))
                else:
                    raise TilusProgramError(self, lhs, "Unexpected assignee: {}".format(type(lookup_result)))
        elif isinstance(lhs, ast.Subscript):
            # example: a[3, 4] = 5.0
            raise NotImplementedError("subscript assignment")
        elif isinstance(lhs, ast.Attribute):
            # example: self.attrs.blocks = 16, 16
            lhs_base = self.visit(lhs.value)

            if lhs_base is self.script.attrs:
                # we only allow the kernel function to assign self.attrs.xxx = ...
                if isinstance(rhs, (tuple, list)):
                    rhs = [hidet_ir.tools.simplify(v) for v in rhs]
                else:
                    rhs = hidet_ir.tools.simplify(rhs)
                setattr(self.script.attrs, lhs.attr, rhs)
            else:
                raise HidetProgramError(self, lhs, "Invalid assignment.")
        else:
            type_name = type(lhs).__name__
            raise HidetProgramError(self, lhs, 'Cannot recognize "{}" as left side of assignment.'.format(type_name))

    def visit_Module(self, module: ast.Module) -> Function:
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            msg = "The module expects to have only one function definition statement, got\n"
            msg += str(ast.unparse(module))
            raise ValueError(msg)
        return self.visit(module.body[0])

    def process_param_ret_type(
        self, arg: ast.AST, arg_type: Union[BaseType, Type[int], Type[float], Type[bool]]
    ) -> BaseType:
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

    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> Function:
        func_params = []
        with self.scope() as scope:
            # process function arguments
            args: ast.arguments = func_def.args

            # make sure that the function parameters only have normal positional arguments
            if args.vararg is not None:
                raise TilusProgramError(self, args.vararg, 'Tilus program does not support "*args" arguments.')
            if args.kwarg is not None:
                raise TilusProgramError(self, args.kwarg, 'Tilus program does not support "**kwargs" arguments.')

            divisibility: dict[Var, int] = {}
            for idx, arg in enumerate(args.args):
                arg_name = arg.arg

                if idx == 0 and arg_name == "self":
                    continue
                if arg_name in self.name2consts:
                    value = self.name2consts[arg_name]
                    self.current_scope.bind(arg_name, value)
                    continue
                if arg_name not in self.type_annotations:
                    raise TilusProgramError(self, arg, "Tilus expects type annotation for each function parameter.")

                arg_type = self.type_annotations[arg_name]
                processed_arg_type: BaseType = self.process_param_ret_type(arg, arg_type)
                param_var = Var(hint=arg_name, type=processed_arg_type)
                func_params.append(param_var)
                scope.bind(arg_name, param_var)
                if arg_name in self.name2divisibility:
                    divisibility[param_var] = self.name2divisibility[arg_name]

            # return type
            if func_def.returns is not None:
                raise TilusProgramError(self, func_def.returns, "Tilus does not support return type annotation.")

            # process function body
            for stmt in func_def.body:
                self.visit(stmt)

            # process the attributes
            attrs = self.script.attrs
            if attrs.blocks is None:
                msg = (
                    "Tilus script should set the number of blocks via self.blocks = ... like\n"
                    "    self.blocks = dim_x\n"
                    "or\n"
                    "    self.blocks = dim_x, dim_y"
                )
                raise TilusProgramError(self, func_def, msg)
            blocks = [as_expr(dim) for dim in normalize_launch_dims(attrs.blocks)]
            attrs.blocks = None

            if attrs.warps is None:
                raise TilusProgramError(
                    self, func_def, "Tilus script should set the number of warps via self.warps = ..."
                )
            warps = attrs.warps
            attrs.warps = None

            return Function.create(
                name=func_def.name,
                params=func_params,
                body=scope.flush_stmts(),
                metadata=Metadata.create(num_blocks=blocks, num_warps=warps, divisibility=divisibility),
            )

    def visit_Expr(self, expr: ast.Expr) -> None:
        value = self.visit(expr.value)

        if value is None:
            # do nothing
            return
        elif isinstance(value, hidet_ir.Expr):
            self.current_scope.append(EvaluateStmt(expr=value, pred=None))
        elif isinstance(value, Tensor):
            # do nothing
            return
        else:
            raise NotImplementedError(value)

    def visit_Call(self, expr: ast.Call) -> Any:
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
            ret = func(*args, **kwargs)
        elif isinstance(func, types.MethodType):
            # call python class method
            ret = func(*args, **kwargs)
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
            ret = func(*args, **kwargs)
        builder_stack: list[list[Stmt]] = self._script._builder._stack
        assert len(builder_stack) == 1
        if len(builder_stack[0]) > 0:
            self.current_scope.stmts.extend(builder_stack[0])
            builder_stack[0].clear()
        return ret

    def visit_Attribute(self, expr: ast.Attribute) -> Any:
        from hidet.ir.primitives.cuda.vars import blockIdx

        base = self.visit(expr.value)
        attr = expr.attr

        if hasattr(base, attr):
            ret = getattr(base, attr)
        else:
            raise HidetProgramError(self, expr, 'Can not access attribute "{}" of object {}.'.format(attr, base))

        self_attributes = {self.script.blockIdx: blockIdx}
        for key in self_attributes:
            if ret is key:
                return self_attributes[key]
        return ret

    def visit_Name(self, expr: ast.Name) -> Any:
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

    def visit_Tuple(self, expr: ast.Tuple) -> Tuple[Any, ...]:
        return tuple(self.visit(v) for v in expr.elts)

    def visit_List(self, expr: ast.List) -> list[Any]:
        return [self.visit(v) for v in expr.elts]

    def visit_BinOp(self, expr: ast.BinOp) -> Union[hidet_ir.Expr, float, int, list, tuple, str]:
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

    def visit_BoolOp(self, expr: ast.BoolOp) -> hidet_ir.Expr:
        values = [self.visit(v) for v in expr.values]
        assert all(isinstance(value, (hidet_ir.Node, bool, int, bool)) for value in values)
        if isinstance(expr.op, ast.And):
            return hidet_ir.logical_and(*values)
        else:
            assert isinstance(expr.op, ast.Or)
            return hidet_ir.logical_or(*values)

    def visit_Assign(self, stmt: ast.Assign) -> None:
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

    def visit_AnnAssign(self, stmt: ast.AnnAssign) -> None:
        lhs = stmt.target
        rhs = self.visit(stmt.value) if stmt.value else None
        assert isinstance(lhs, (ast.Name, ast.Attribute, ast.Subscript))
        if isinstance(lhs, (ast.Attribute, ast.Subscript)):
            msg = 'Hidet do not support annotation for expression like "x.y" or "x[y]"'
            raise HidetProgramError(self, stmt.annotation, msg)
        self.process_assign(lhs, rhs, stmt.annotation)

    def visit_Lambda(self, expr: ast.Lambda) -> LambdaProxy:
        return LambdaProxy(expr, self)

    def visit_Subscript(self, expr: ast.Subscript) -> Any:
        base = self.visit(expr.value)
        indices = self.visit(expr.slice)

        if isinstance(base, Sequence):
            return base[indices]
        else:
            raise NotImplementedError()

    def visit_Constant(self, expr: ast.Constant) -> Union[float, int, str, None]:
        if isinstance(expr.value, (float, int)):
            return expr.value
        elif isinstance(expr.value, str):
            return expr.value
        elif expr.value is None:
            return expr.value
        else:
            raise HidetProgramError(self, expr, "Can not recognize Constant {}".format(repr(expr.value)))

    def visit_Compare(self, expr: ast.Compare) -> hidet_ir.Expr:
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
                cur_cond: Any = op_dict[op_kind](front, current)  # type: ignore[operator]
            else:
                cur_cond = py_op_dict[op_kind](front, current)
            cond = hidet_ir.logical_and(cond, cur_cond) if cond is not None else cur_cond
            front = current
        assert isinstance(cond, hidet_ir.Expr)
        return cond

    def visit_For(self, stmt: ast.For) -> None:
        # create loop vars
        iter_targets: list[ast.Name] = []
        if isinstance(stmt.target, (ast.List, ast.Tuple)):
            for target in stmt.target.elts:
                if not isinstance(target, ast.Name):
                    raise HidetProgramError(self, stmt, "For loop target must be a name.")
                iter_targets.append(target)
        else:
            if not isinstance(stmt.target, ast.Name):
                raise HidetProgramError(self, stmt, "For loop target must be a name.")
            iter_targets.append(stmt.target)

        # construct for body
        stmt_iter = self.visit(stmt.iter)
        num_targets: int = len(iter_targets)
        if isinstance(stmt_iter, TilusLoopIterable):
            loop_vars: list[Var] = []
            host_vars: dict[str, Any] = {}

            num_loop_vars: int = stmt_iter.num_loop_vars()

            if num_targets == num_loop_vars > 1 or (num_targets == num_loop_vars == 1 and not stmt_iter.bind_tuple()):
                for target in iter_targets:
                    loop_vars.append(Var(target.id, type=hidet_ir.data_type("int32")))
            elif num_targets == 1:
                name = iter_targets[0].id
                for i in range(num_loop_vars):
                    loop_vars.append(Var(f"{name}{i}", type=hidet_ir.data_type("int32")))
                host_vars[name] = list(loop_vars)
            else:
                raise HidetProgramError(
                    self, stmt, f"Expect {num_loop_vars} loop variables, but got {len(iter_targets)}."
                )

            with self.scope() as for_scope:
                for var in loop_vars:
                    assert var.hint is not None
                    for_scope.bind(name=var.hint, var_or_value=var)
                for name, value in host_vars.items():
                    for_scope.bind(name, value)
                for s in stmt.body:
                    self.visit(s)
            body = for_scope.flush_stmts()
            for_stmt = stmt_iter.generate_loop_statement(loop_vars=loop_vars, body=body)
            self.current_scope.append(for_stmt)
        else:
            msg = "For loop iterable must be a one of the following types: \n1.\n  for ... in range(...): \n      ...\n"
            raise HidetProgramError(self, stmt.iter, msg)
