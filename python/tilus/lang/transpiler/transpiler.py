# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import ast
import builtins
import inspect
import math
import operator
import types
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

from hidet import ir as hidet_ir
from hidet.ir.expr import Constant, Var, as_expr
from hidet.ir.primitives.cuda.vars import blockIdx
from hidet.ir.type import BaseType, data_type
from hidet.lang.script import eliminate_decorators, eliminate_indent
from hidet.lang.transpiler import PythonAstFunctor

from tilus import ir as tilus_ir
from tilus.extensions.hidet.ir.tools.type_infer import infer_type
from tilus.ir.func import Function, Metadata
from tilus.ir.inst import InstructionError
from tilus.ir.stmt import DeclareStmt, SeqStmt, Stmt
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.ir.utils import frozendict
from tilus.ir.utils.normalize import normalize_cluster_blocks, normalize_grid_blocks
from tilus.lang.constructs import State
from tilus.lang.constructs.contexts import TilusContext
from tilus.lang.constructs.loops import TilusLoopIterable
from tilus.lang.instructions import builder_context
from tilus.lang.methods import (
    GlobalTensorWithMethods,
    RegisterTensorWithMethods,
    SharedTensorWithMethods,
    TensorMethodError,
)
from tilus.lang.script import Attributes, Script

from .builder import ScopedProgramBuilder
from .common import TilusProgramError


class LambdaProxy:
    """A proxy for lambda function defined in Tilus Script."""

    def __init__(self, lambda_expr: ast.Lambda, translator: Transpiler):
        self.lambda_expr: ast.Lambda = lambda_expr
        self.translator: Transpiler = translator

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TilusProgramError(
                self.translator, self.lambda_expr, "Do not support keyword arguments in lambda function."
            )

        with self.translator.scope() as lambda_params_scope:
            if len(args) != len(self.lambda_expr.args.args):
                raise TilusProgramError(
                    self.translator,
                    self.lambda_expr,
                    "The number of arguments does not match the lambda function definition.",
                )
            for arg, arg_expr in zip(self.lambda_expr.args.args, args):
                arg_name = arg.arg
                lambda_params_scope.bind(arg_name, arg_expr)
            return self.translator.visit(self.lambda_expr.body)


class Transpiler(ScopedProgramBuilder, PythonAstFunctor):
    def __init__(self) -> None:
        PythonAstFunctor.__init__(self, file="", start_lineno=0, start_column=0)  # type: ignore[call-arg]
        ScopedProgramBuilder.__init__(self)
        self._optional_script: Optional[Script] = None

    def visit(self, node):
        """dispatch method for visiting an AST node."""

        method = "visit_" + node.__class__.__name__
        if hasattr(self, method):
            visitor = getattr(self, method)
        else:
            msg = "The AST node {} is not supported in HidetScript.".format(node.__class__.__name__)
            raise TilusProgramError(self, node, msg)

        try:
            return visitor(node)
        except TilusProgramError as e:
            if e.obj is None and isinstance(node, (ast.stmt, ast.expr)):
                e.obj = node
            raise
        except InstructionError as e:
            raise TilusProgramError(self, node, str(e)) from e
        except Exception as e:
            raise TilusProgramError(self, node, "Internal exception occurred during transpiling this ast node.") from e

    @property
    def script(self) -> Script:
        if self._optional_script is None:
            raise RuntimeError("The script is not set.")
        return self._optional_script

    def create_script_call_args(
        self,
        script: Script,
        name2consts: dict[str, int | float | str],
    ) -> dict[str, Script | State | Var | int | float | str | bool]:
        inspect_sig: inspect.Signature = inspect.signature(script.__class__.__call__)
        params: dict[str, Script | State | Var | int | float | str | bool] = {}
        for idx, (name, param) in enumerate(inspect_sig.parameters.items()):
            # self parameter
            if idx == 0:
                params[name] = script
                continue

            # constant parameter
            if name in name2consts:
                params[name] = name2consts[name]
                continue

            # all other parameters will be the parameters of the kernel function
            # it must have type annotation
            if param.annotation is inspect.Parameter.empty:
                raise TilusProgramError(
                    self,
                    None,
                    f'Parameter "{name}" has no type annotation, tilus requires type annotation for each parameter.',
                )
            param_type = param.annotation
            if not isinstance(param_type, BaseType):
                raise TilusProgramError(
                    self,
                    None,
                    f'Parameter "{name}" has invalid type annotation "{param_type}", tilus requires a BaseType type annotation for each parameter.',
                )
            param_var = Var(hint=name, type=param_type)
            params[name] = param_var

        # return type
        if inspect_sig.return_annotation is not inspect.Signature.empty:
            raise TilusProgramError(
                self,
                None,
                "Tilus does not support return type annotation for __call__ method, got {}".format(
                    inspect_sig.return_annotation
                ),
            )
        return params

    def transpile(
        self, script: Script, name2consts: dict[str, Union[int, float, str, Any]], name2divisibility: dict[str, int]
    ) -> Function:
        """
        Transpile the __call__(...) method of the given Tilus Script into a Tilus Function.

        This method transpiles the __call__(...) method of the given Tilus Script into a Tilus Function. It
        extracts the source code of the __call__ method, parses it into an AST, and then visits the AST nodes to
        generate the corresponding Tilus IR. The resulting Tilus Function represents the computation defined in the
        __call__ method.

        The `name2consts` parameter is a dictionary that maps variable names to constant values. All parameters with
        type annotations int, float, or str are compilation-time constants, and different values will trigger Just-In-Time
        (JIT) recompilation of the Tilus Script. This parameter provides the mapping from variable names to their constant
        values during one JIT compilation.

        The `name2divisibility` parameter is a dictionary that maps variable names to their divisibility requirements.
        Parameters with integer type annotations (e.g., tilus.int32, tilus.int64, etc.) can have different divisibility.
        Different divisibility will also trigger JIT recompilation of the Tilus Script. This parameter provides the mapping
        from variable names to their divisibility during one JIT compilation.
        """
        # declare or bind the parameters of the __call__ method
        params: dict[str, Script | State | Var | int | float | str | bool] = self.create_script_call_args(
            script, name2consts
        )
        kernel_params: tuple[Var, ...] = tuple(param for param in params.values() if isinstance(param, Var))

        # transpile the __call__ method body
        with builder_context(self):
            self.transpile_call(script.__class__.__call__, params.values(), {})
        body: Stmt = self.flush_stmts()

        # check and create metadata
        if script.attrs.blocks is None:
            raise RuntimeError("The script.attrs.blocks is not set.")
        if script.attrs.warps is None:
            raise RuntimeError("The script.attrs.warps is not set.")
        param2divisibility: dict[Var, int] = {}
        for name in name2divisibility:
            param = params[name]
            assert isinstance(param, Var)
            param2divisibility[param] = name2divisibility[name]
        metadata = Metadata(
            grid_blocks=normalize_grid_blocks(script.attrs.blocks),
            cluster_blocks=normalize_cluster_blocks(script.attrs.cluster_blocks),
            block_indices=(blockIdx.x, blockIdx.y, blockIdx.z),
            num_warps=script.attrs.warps,
            param2divisibility=frozendict(param2divisibility),
            analysis=None,
        )

        func = Function(name=script.__class__.__name__, params=kernel_params, body=body, metadata=metadata)

        return func

    def get_external_env(self, func: Union[types.FunctionType, types.MethodType]) -> dict[str, Any]:
        # Get the environment (globals and binding of free variables)
        # See the data model of python for the details of func.__globals__, func.__closure__ and func.__code__:
        #     https://docs.python.org/3/reference/datamodel.html
        if isinstance(func, types.MethodType):
            assert isinstance(func.__func__, types.FunctionType)
            func = func.__func__
        env: dict[str, Any] = func.__globals__.copy()
        func_freevar_names: list[str] = list(func.__code__.co_freevars)
        func_freevar_cells: list[Any] = [v.cell_contents for v in func.__closure__] if func.__closure__ else []
        assert len(func_freevar_names) == len(func_freevar_cells)
        env.update(dict(zip(func_freevar_names, func_freevar_cells)))
        return env

    def transpile_call(self, func, args, kwargs):
        sig: inspect.Signature = inspect.signature(func)
        bound_args: inspect.BoundArguments = sig.bind(*args, **kwargs)

        ret = None

        # we need to dump the current scopes and push it to a stack
        self.dump_and_push_scopes()

        with self.scope():  # external scope
            # bind the external environment in a new scope
            external_env = self.get_external_env(func)
            for name, value in external_env.items():
                self.bind(name, value)

            with self.scope():  # parameter scope
                # bind the parameters to the arguments in a new scope
                for idx, param_name in enumerate(sig.parameters):
                    param: inspect.Parameter = sig.parameters[param_name]
                    arg = bound_args.arguments[param_name]
                    annotation = param.annotation

                    if idx == 0:
                        # the self parameter, it does not have type annotation, treat it as host_var
                        self.bind(param_name, arg)
                    elif annotation is inspect.Parameter.empty:
                        # for all other parameters, it must have type annotation to be executed in transpile-mode
                        raise TilusProgramError(self, None, 'Parameter "{}" has no type annotation.'.format(param_name))
                    elif isinstance(annotation, str):
                        # this usually happens when "from __future__ import annotations" is used
                        # in such case, the annotation is stored as a string
                        # we currently do not support this feature
                        raise TilusProgramError(
                            self,
                            None,
                            (
                                "A python string as parameter type annotation detected. \n"
                                'This is usually because "from __future__ import annotations" has been used.\n'
                                "Currently, tilus script is not compatible with this feature. \n"
                                "Please considering not using it in module that defines tilus script."
                            ),
                        )
                    elif annotation in (RegisterTensor, SharedTensor, GlobalTensor):
                        # tensor parameters are passed by reference
                        if not isinstance(arg, annotation):
                            raise TilusProgramError(
                                self,
                                None,
                                'Parameter "{}" expects a {} but got {}.'.format(
                                    param_name, annotation.__name__, type(arg).__name__
                                ),
                            )
                        self.bind(param_name, arg)
                    elif isinstance(annotation, (hidet_ir.DataType, hidet_ir.PointerType)):
                        # scalar parameters are passed by value
                        if not isinstance(arg, (hidet_ir.Expr, int, bool, float)):
                            raise TilusProgramError(
                                self,
                                None,
                                'Parameter "{}" expects an expression but got {}.'.format(
                                    param_name, type(arg).__name__
                                ),
                            )
                        var = self.declare(type=annotation, init=as_expr(arg))
                        self.bind(param_name, var)
                    elif annotation in [bool, int, float]:
                        # python built-in constants are passed by value, but we don't need to declare a variable for them
                        # since they are immutable
                        if not isinstance(arg, Constant) and not isinstance(arg, annotation):
                            raise TilusProgramError(
                                self,
                                None,
                                'Parameter "{}" expects a constant but got {}.'.format(param_name, type(arg).__name__),
                            )
                        self.bind(param_name, annotation(arg))
                    elif inspect.isclass(annotation) and issubclass(annotation, State):
                        # State are passed by reference
                        state_cls: Type[State] = annotation
                        if not isinstance(arg, state_cls):
                            raise TilusProgramError(
                                self,
                                None,
                                'Parameter "{}" expects a {} but got {}.'.format(
                                    param_name, state_cls.__name__, type(arg).__name__
                                ),
                            )
                        self.bind(param_name, arg)
                    else:
                        # unsupported type annotation
                        raise TilusProgramError(
                            self,
                            None,
                            'Parameter "{}" has an unsupported type annotation: {}.\n'.format(param_name, annotation)
                            + "Currently, we only support data type, pointer, and tensors as type annotations.",
                        )

                # process the body, we transpile-run the function body in the new scope in the current scope stack,
                # instead of creating a new Function. It implements the inlined kernel procedure feature.
                lines, start_line = inspect.getsourcelines(func)
                file: Optional[str] = inspect.getsourcefile(func)
                if file is None:
                    raise RuntimeError('Can not get the source file of the given function "{}".'.format(func.__name__))

                source = "".join(lines)
                source, col_offset = eliminate_indent(source)
                source, inc_lineno = eliminate_decorators(source)
                start_line += inc_lineno
                parsed: ast.Module = ast.parse(source=source)
                func_defs = parsed.body
                assert len(func_defs) == 1 and isinstance(func_defs[0], ast.FunctionDef)
                func_def: ast.FunctionDef = func_defs[0]

                old = self.file, self.start_lineno, self.start_column  # type: ignore[has-type]
                self.file, self.start_lineno, self.start_column = file, start_line, col_offset
                with self.scope():  # body scope
                    for i, stmt in enumerate(func_def.body):
                        if isinstance(stmt, ast.Return):
                            if i != len(func_def.body) - 1:
                                raise TilusProgramError(
                                    self, stmt, "Return statement must be the last statement in a tilus procedure."
                                )
                            ret = self.visit(stmt.value)
                            continue
                        self.visit(stmt)
                self.file, self.start_lineno, self.start_column = old

        # after the function call, pop and restore the caller's scopes
        self.pop_and_restore_scopes()

        return ret

    def assign_value_to_var(self, var: Union[Var, Tensor], value: Any) -> None:
        if isinstance(var, Var):
            if not isinstance(value, (hidet_ir.Expr, int, float, str)):
                raise TilusProgramError(self, None, "Assignment between Var is only accepted for hidet_ir.Expr.")
            self.assign(var=var, value=as_expr(value))
        elif isinstance(var, Tensor):
            if not isinstance(value, RegisterTensor) or not isinstance(var, RegisterTensor):
                raise TilusProgramError(self, None, "Assignment between Value is only accepted for RegisterValue.")
            from hidet.ir.type import type_equal

            if not type_equal(var.dtype, value.dtype):
                raise TilusProgramError(
                    self,
                    None,
                    "Different types of RegisterValue are not allowed to be assigned to each other. ",
                )
            self.assign_register(output=var, x=value)
        else:
            assert False

    def process_name_assign(self, name: str, rhs: Any, type_annotation: Optional[ast.expr] = None) -> None:
        if name == "_":
            # discard the assignment to '_' variable
            return
        bound_value = self.lookup(name)
        value = rhs
        if bound_value is None:
            # bind a new name to the right side, the rhs could be
            #  1) a hidet expression => we define a new scalar variable
            #  2) a tilus value => we bind the value to the name
            #  3) other host expressions
            #    3.1) if there is type annotation, we define a scalar variable
            #    3.2) otherwise, we bind the host expression to the name
            if isinstance(value, hidet_ir.Expr):
                var = self.declare(type=hidet_ir.infer_type(value), init=value, hint=name)
                self.bind(name, var)
            elif isinstance(value, tilus_ir.Tensor):
                self.bind(name, value)
            else:
                if value is None:
                    raise TilusProgramError(self, None, "Cannot bind or assign None value.")
                if type_annotation is not None:
                    resolved_annotation = self.visit(type_annotation)
                    if resolved_annotation in (int, str, float):
                        value = resolved_annotation(value)  # type: ignore
                        self.bind(name, value)
                    else:
                        if not isinstance(resolved_annotation, (hidet_ir.DataType, hidet_ir.PointerType)):
                            raise TilusProgramError(
                                self, type_annotation, "Invalid type annotation: {}".format(resolved_annotation)
                            )
                        if not isinstance(value, hidet_ir.Expr):
                            value = as_expr(value)  # type: ignore
                        stmt = DeclareStmt(var=Var(hint=name, type=resolved_annotation), init=value)
                        self.append(stmt)
                        self.bind(name, stmt.var)
                else:
                    if isinstance(value, State):
                        # we allow to bind State without type annotation
                        self.bind(name, value)
                    else:
                        raise TilusProgramError(self, None, "Cannot bind or assign value without type annotation.")
        else:
            # assignment
            if isinstance(bound_value, (Var, Tensor)):
                self.assign_value_to_var(bound_value, value)
            else:
                raise TilusProgramError(self, None, "Unexpected assignee: {}".format(type(bound_value)))

    def process_subscript_assign(
        self, lhs: ast.Subscript, rhs: Any, type_annotation: Optional[ast.expr] = None
    ) -> None:
        # example: a[3, 4] = 5.0
        dst_tensor = self.visit(lhs.value)
        if not isinstance(dst_tensor, RegisterTensor):
            raise TilusProgramError(self, lhs, "The left side of subscript assignment must be a RegisterTensor.")

        # extract offsets and slice_dims
        indices = self.visit(lhs.slice)
        offsets = []
        slice_dims = []

        if not isinstance(indices, Sequence):
            indices = [indices]
        else:
            indices = list(indices)
        while len(indices) < len(dst_tensor.shape):
            indices.append(slice(None, None, None))
        for dim, index in enumerate(indices):
            if isinstance(index, slice):
                if index.start is not None:
                    offsets.append(as_expr(index.start))
                else:
                    offsets.append(Constant(0, data_type("int32")))
                slice_dims.append(dim)
            else:
                offsets.append(as_expr(index))

        # process rhs
        if isinstance(rhs, RegisterTensor):
            rhs_tensor = rhs
        else:
            rhs_tensor = self.allocate_register(
                dtype=dst_tensor.dtype, shape=[], f_init=lambda _: dst_tensor.dtype(rhs)
            )

        self.slice_assign_register(output=dst_tensor, x=rhs_tensor, offsets=offsets, dims=slice_dims)

    def get_attribute_chain(self, node: ast.Attribute) -> list[str]:
        if isinstance(node.value, ast.Attribute):
            attrs = self.get_attribute_chain(node.value)
            attrs.append(node.attr)
            return attrs
        elif isinstance(node.value, ast.Name):
            return [node.value.id, node.attr]
        else:
            raise TilusProgramError(self, node, "Invalid attribute access: {}".format(ast.dump(node)))

    def process_attribute_assign(
        self, lhs: ast.Attribute, rhs: Any, type_annotation: Optional[ast.expr] = None
    ) -> None:
        # example: self.attrs.blocks = 16, 16
        lhs_base = self.visit(lhs.value)

        if isinstance(lhs_base, Attributes):
            setattr(lhs_base, lhs.attr, rhs)
        elif isinstance(lhs_base, State):
            # State.xxx = ...
            if hasattr(lhs_base, lhs.attr):
                # we have defined the attribute, assign it
                self.assign_value_to_var(getattr(lhs_base, lhs.attr), rhs)
            else:
                name = "_".join(self.get_attribute_chain(lhs))
                self.process_name_assign(name=name, rhs=rhs, type_annotation=type_annotation)
                setattr(lhs_base, lhs.attr, self.lookup(name))
        else:
            raise TilusProgramError(self, lhs, "Invalid assignment: {}".format(type(lhs_base)))

    def process_assign(
        self, lhs: Union[ast.Attribute, ast.Subscript, ast.Name], rhs: Any, type_annotation: Optional[ast.expr] = None
    ) -> None:
        # three cases of assignment:
        #    1. v = ...
        #    2. a[i, j] = ...
        #    3. attr.name = ...
        if isinstance(lhs, ast.Name):
            self.process_name_assign(lhs.id, rhs, type_annotation)
        elif isinstance(lhs, ast.Subscript):
            self.process_subscript_assign(lhs, rhs, type_annotation)
        elif isinstance(lhs, ast.Attribute):
            self.process_attribute_assign(lhs, rhs, type_annotation)
        else:
            assert False

    def visit_Expr(self, expr: ast.Expr) -> None:
        value = self.visit(expr.value)

        if value is None:
            # do nothing
            return
        elif isinstance(value, hidet_ir.Expr):
            self.evaluate(pred=None, expr=value)
        elif isinstance(value, Tensor):
            # do nothing
            return
        elif isinstance(value, str):
            # doc string, do nothing
            return
        else:
            raise NotImplementedError(value)

    def visit_Call(self, expr: ast.Call) -> Any:
        # prepare the func, args, and kwargs for the function call
        #   func(*args, **kwargs)
        func = self.visit(expr.func)
        args: list[Any] = []
        for arg in expr.args:
            if isinstance(arg, ast.Starred):
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
        kwargs: dict[str, Any]
        if len(expr.keywords) == 0:
            kwargs = {}
        elif len(expr.keywords) == 1 and expr.keywords[0].arg is None:
            # func(a, b, **kwargs)
            kwargs = self.visit(expr.keywords[0].value)
        else:
            # func(a=1, b=2, c=3)
            if any(kwarg.arg is None for kwarg in expr.keywords):
                raise TilusProgramError(self, expr, "Mixing of keyword arguments and **kwargs is not supported.")
            kwargs = {kwarg.arg: self.visit(kwarg.value) for kwarg in expr.keywords if kwarg.arg is not None}

        try:
            """
            There are different kinds of function calls in Tilus Script:
            1. inlined kernel procedure, it is a method of the user-defined Script subclass, or a user-defined State subclass.
            2. (global, shared or register) tensor method, such as `tensor.to(dtype)`, etc.
            3. python builtin function, such as `max`, `min`, for scalar expressions.
            4. subclass of tilus.State
            5. other function/method calls

            We treat 1 to 3 specially, and call the function directly in 4.
            """

            if isinstance(func, types.MethodType):
                f_self = func.__self__
                f_func = func.__func__
                if isinstance(f_self, Script) and getattr(Script, f_func.__name__, None) is not f_func:
                    # case 1 (Script method)
                    ret = self.transpile_call(f_func, [f_self, *args], kwargs)
                elif isinstance(f_self, State) and getattr(State, f_func.__name__, None) is not f_func:
                    # case 1 (State method)
                    ret = self.transpile_call(f_func, [f_self, *args], kwargs)
                elif isinstance(f_self, (GlobalTensor, SharedTensor, RegisterTensor)):
                    # case 2
                    method_name = func.__name__
                    tensor_with_methods: RegisterTensorWithMethods | SharedTensorWithMethods | GlobalTensorWithMethods
                    if isinstance(f_self, RegisterTensor):
                        tensor_with_methods = RegisterTensorWithMethods(f_self, self)
                    elif isinstance(f_self, SharedTensor):
                        tensor_with_methods = SharedTensorWithMethods(f_self, self)
                    elif isinstance(f_self, GlobalTensor):
                        tensor_with_methods = GlobalTensorWithMethods(f_self, self)
                    else:
                        raise NotImplementedError(
                            "Currently, only RegisterTensor methods are supported in Tilus Script."
                        )
                    if not hasattr(tensor_with_methods, method_name):
                        raise TilusProgramError(
                            self, expr, 'Method "{}" is not found in {}.'.format(method_name, type(f_self).__name__)
                        )
                    try:
                        ret = getattr(tensor_with_methods, method_name)(*args, **kwargs)
                    except TensorMethodError as e:
                        raise TilusProgramError(self, expr, str(e))
                else:
                    # case 4
                    try:
                        ret = func(*args, **kwargs)
                    except TypeError as e:
                        raise TilusProgramError(self, expr, str(e)) from e
            elif isinstance(func, (types.BuiltinMethodType, types.BuiltinFunctionType)):
                # case 3
                from hidet import ir
                from hidet.ir import primitives

                if all(not isinstance(arg, ir.Node) for arg in args):
                    # pure python function call
                    ret = func(*args, **kwargs)
                else:
                    if any(not isinstance(arg, (ir.Expr, int, float, bool)) for arg in args):
                        # if any argument is not a valid expression
                        ret = func(*args, **kwargs)
                    else:
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
                            raise TilusProgramError(self, expr, msg)
                        if func in func_map:
                            arity, hidet_func = func_map[func]  # type: ignore[index]
                            if len(args) != arity:
                                msg = f'Hidet builtin function "{func.__name__}" takes {arity} arguments.'
                                raise TilusProgramError(self, expr, msg)
                            ret = hidet_func(*args)  # type: ignore[operator]
                        else:
                            raise TilusProgramError(
                                self,
                                expr,
                                'Currently, do not support calling python builtin function "{}".'.format(
                                    func.__qualname__
                                ),
                            )
            elif inspect.isclass(func) and issubclass(func, State):
                # case 4
                state_cls: Type[State] = func
                state_obj = object.__new__(state_cls)
                self.transpile_call(state_cls.__init__, [state_obj, *args], kwargs)
                ret = state_obj
            else:
                # case 5
                ret = func(*args, **kwargs)

            return ret
        except InstructionError as e:
            raise TilusProgramError(self, expr, str(e)) from e

    def visit_Attribute(self, expr: ast.Attribute) -> Any:
        base = self.visit(expr.value)
        attr = expr.attr
        if hasattr(base, attr):
            ret = getattr(base, attr)
        else:
            raise TilusProgramError(self, expr, 'Can not access attribute "{}" of object {}.'.format(attr, base))
        return ret

    def visit_Name(self, expr: ast.Name) -> Any:
        if isinstance(expr.ctx, ast.Store):
            raise ValueError("Internal Error, please deal with all Store behavior in parent nodes like Assign.")
        elif isinstance(expr.ctx, ast.Load):
            name: str = expr.id
            var = self.lookup(name)
            if var is None:
                if name in builtins.__dict__:
                    # access builtin functions such as max, min
                    return getattr(builtins, name)
                raise TilusProgramError(self, expr, "Trying to access variable without definition: {}".format(name))
            return var
        elif isinstance(expr.ctx, ast.Del):
            raise TilusProgramError(self, expr, "Hidet does not support del statement.")
        else:
            raise ValueError()

    def visit_Tuple(self, expr: ast.Tuple) -> Tuple[Any, ...]:
        return tuple(self.visit(v) for v in expr.elts)

    def visit_List(self, expr: ast.List) -> list[Any]:
        return [self.visit(v) for v in expr.elts]

    def visit_BinOp(self, expr: ast.BinOp) -> Union[hidet_ir.Expr, RegisterTensor, float, int, list, tuple, str]:
        from hidet import ir

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
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
        }

        lhs = self.visit(expr.left)
        rhs = self.visit(expr.right)
        if isinstance(lhs, str) and isinstance(rhs, str):
            assert isinstance(expr.op, ast.Add)
            return lhs + rhs
        elif isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
            assert isinstance(expr.op, ast.Add)
            return list(lhs) + list(rhs)
        elif isinstance(lhs, (ir.Expr, float, int)) and isinstance(rhs, (ir.Expr, float, int)):
            if type(expr.op) in op_dict:
                return op_dict[type(expr.op)](lhs, rhs)
            else:
                type_name = type(expr.op).__name__
                raise TilusProgramError(self, expr, "Currently, we do not support {} operator.".format(type_name))
        elif isinstance(lhs, RegisterTensor) or isinstance(rhs, RegisterTensor):
            if not isinstance(lhs, RegisterTensor):
                lhs = self.allocate_register(dtype=rhs.dtype, shape=rhs.shape, f_init=lambda _: rhs.dtype(lhs))
            if not isinstance(rhs, RegisterTensor):
                rhs = self.allocate_register(dtype=lhs.dtype, shape=lhs.shape, f_init=lambda _: lhs.dtype(rhs))

            if isinstance(lhs, RegisterTensor):
                lhs = RegisterTensorWithMethods(lhs, self)
            if isinstance(rhs, RegisterTensor):
                rhs = RegisterTensorWithMethods(rhs, self)

            if type(expr.op) in op_dict:
                return op_dict[type(expr.op)](lhs, rhs)
            else:
                type_name = type(expr.op).__name__
                raise TilusProgramError(self, expr, "Currently, we do not support {} operator.".format(type_name))
        else:
            raise TilusProgramError(
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
            raise TilusProgramError(self, stmt, 'Hidet does not support syntax like "a = b = 1".')
        target = stmt.targets[0]
        value = stmt.value

        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            # a, b = c, d
            lhs_list = target.elts
            rhs_list = [self.visit(v) for v in value.elts]
            if len(lhs_list) != len(rhs_list):
                raise TilusProgramError(self, stmt, "The number of left values and right values does not match.")
            for lhs, rhs in zip(lhs_list, rhs_list):
                assert isinstance(lhs, (ast.Attribute, ast.Subscript, ast.Name))
                self.process_assign(lhs, rhs)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # a, b = c
            lhs_list = target.elts
            rhs_list = self.visit(value)
            if len(lhs_list) != len(rhs_list):
                raise TilusProgramError(self, stmt, "The number of left values and right values does not match.")
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
        self.process_assign(lhs, rhs, type_annotation=stmt.annotation)

    def visit_Lambda(self, expr: ast.Lambda) -> LambdaProxy:
        return LambdaProxy(expr, self)

    def visit_Subscript(self, expr: ast.Subscript) -> Any:
        base = self.visit(expr.value)
        indices = self.visit(expr.slice)

        if isinstance(base, Sequence):
            return base[indices]
        elif isinstance(base, (GlobalTensor, SharedTensor, RegisterTensor)):
            if not isinstance(indices, Sequence):
                indices = [indices]
            else:
                indices = list(indices)
            offsets = []
            slice_dims = []
            for dim, idx in enumerate(indices):
                if isinstance(idx, slice):
                    if idx.start is not None or idx.stop is not None:
                        if not isinstance(idx.start, (int, hidet_ir.Expr)):
                            raise TilusProgramError(
                                self,
                                expr,
                                "Global/Shared tensors only support slicing whole dimensions: [..., :, ...], "
                                "do not support slicing like [..., start:, ...] or [..., :end, ...].",
                            )
                    offsets.append(0)
                    slice_dims.append(dim)
                else:
                    offsets.append(idx)

            if len(offsets) < len(base.shape):
                dim = len(offsets)
                while len(offsets) < len(base.shape):
                    offsets.append(0)
                    slice_dims.append(dim)
                    dim += 1
            if len(indices) > len(base.shape):
                raise TilusProgramError(self, expr, "Too many indices for tensor of shape {}.".format(base.shape))

            sliced_tensor: Union[GlobalTensor, SharedTensor, RegisterTensor]
            if isinstance(base, GlobalTensor):
                sliced_tensor = self.slice_global(
                    tensor=base,
                    offsets=offsets,
                    slice_dims=slice_dims,
                    slice_shape=[base.shape[dim] for dim in slice_dims],
                )
            elif isinstance(base, SharedTensor):
                sliced_tensor = self.slice_shared(
                    tensor=base,
                    offsets=offsets,
                    slice_dims=slice_dims,
                    slice_shape=[base.shape[dim] for dim in slice_dims],
                )
            elif isinstance(base, RegisterTensor):
                sliced_tensor = self.slice_register(
                    tensor=base,
                    offsets=offsets,
                    slice_dims=slice_dims,
                    slice_shape=[base.shape[dim] for dim in slice_dims],
                )
            else:
                assert False
            return sliced_tensor
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
            raise TilusProgramError(self, expr, "Can not recognize Constant {}".format(repr(expr.value)))

    def visit_Compare(self, expr: ast.Compare) -> Union[hidet_ir.Expr, RegisterTensor]:
        operands = [self.visit(expr.left)] + [self.visit(v) for v in expr.comparators]

        if any(isinstance(operand, RegisterTensor) for operand in operands):
            operands = [
                operand
                if isinstance(operand, RegisterTensor)
                else self.allocate_register(dtype=infer_type(operand), shape=[], f_init=lambda axes: operand)
                for operand in operands
            ]
            op_dict = {
                ast.Eq: self.equal,
                ast.NotEq: self.not_equal,
                ast.Gt: self.greater_than,
                ast.Lt: self.less_than,
                ast.GtE: self.greater_equal,
                ast.LtE: self.less_equal,
            }
            left = operands.pop(0)
            for op, right in zip(expr.ops, operands):
                if type(op) not in op_dict:
                    raise TilusProgramError(self, expr, "Currently, we do not support {} operator.".format(type(op)))
                left = op_dict[type(op)](left, right)
            return left
        else:
            operands = [as_expr(operand) for operand in operands]
            op_dict: Any = {  # type: ignore[no-redef]
                ast.Eq: hidet_ir.equal,
                ast.NotEq: hidet_ir.not_equal,
                ast.Gt: lambda a, b: hidet_ir.less_than(b, a),  # pylint: disable=arguments-out-of-order
                ast.Lt: hidet_ir.less_than,
                ast.GtE: lambda a, b: hidet_ir.less_equal(b, a),  # pylint: disable=arguments-out-of-order
                ast.LtE: hidet_ir.less_equal,
            }
            left = operands.pop(0)
            for op, right in zip(expr.ops, operands):
                if type(op) not in op_dict:
                    raise TilusProgramError(self, expr, "Currently, we do not support {} operator.".format(type(op)))
                left = op_dict[type(op)](left, right)
            return left

    def visit_IfExp(self, expr: ast.IfExp) -> hidet_ir.Expr:
        cond = self.visit(expr.test)

        if isinstance(cond, hidet_ir.Constant) or isinstance(cond, (int, bool)):
            cond = bool(cond)
            if cond:
                then_expr = self.visit(expr.body)
                return then_expr
            else:
                else_expr = self.visit(expr.orelse)
                return else_expr
        else:
            then_expr = self.visit(expr.body)
            else_expr = self.visit(expr.orelse)
            if not isinstance(then_expr, (hidet_ir.Expr, int, bool, float)) or not isinstance(
                else_expr, (hidet_ir.Expr, int, bool, float)
            ):
                raise TilusProgramError(self, expr, "Then and else expression must be hidet expression.")
            return hidet_ir.expr.if_then_else(cond, then_expr, else_expr)

    def visit_AugAssign(self, stmt: ast.AugAssign) -> None:
        if isinstance(stmt.target, ast.Name):
            target = ast.Name(stmt.target.id, ast.Load())
            var_value = self.visit(target)
            value = self.visit(stmt.value)

            if isinstance(var_value, RegisterTensor):
                if isinstance(value, (int, float, hidet_ir.Expr)):
                    value = self.allocate_register(
                        dtype=var_value.dtype, shape=var_value.shape, f_init=lambda _: var_value.dtype(value)
                    )
                if isinstance(stmt.op, ast.Add):
                    self.add(x=var_value, y=value, out=var_value)
                elif isinstance(stmt.op, ast.Sub):
                    self.sub(x=var_value, y=value, out=var_value)
                elif isinstance(stmt.op, ast.Mult):
                    self.mul(x=var_value, y=value, out=var_value)
                elif isinstance(stmt.op, ast.Div):
                    self.div(x=var_value, y=value, out=var_value)
                elif isinstance(stmt.op, ast.FloorDiv):
                    self.div(x=var_value, y=value, out=var_value)
                elif isinstance(stmt.op, ast.Mod):
                    self.mod(x=var_value, y=value, out=var_value)
                else:
                    raise TilusProgramError(self, stmt, "AugAssign only support RegisterTensor or hidet expression.")
            elif isinstance(var_value, hidet_ir.Var) and isinstance(value, (int, float, hidet_ir.Expr)):
                op_dict = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Mod: operator.mod,
                    ast.RShift: operator.rshift,
                    ast.LShift: operator.lshift,
                    ast.BitXor: operator.xor,
                }
                self.assign(var=var_value, value=op_dict[type(stmt.op)](var_value, value))
            else:
                raise TilusProgramError(self, stmt, "AugAssign only support RegisterTensor or hidet expression.")
        elif isinstance(stmt.target, ast.Subscript):
            # example: a[3, 4] = 5.0
            dst_tensor = self.visit(stmt.target.value)
            if not isinstance(dst_tensor, RegisterTensor):
                raise TilusProgramError(
                    self, stmt.target, "The left side of subscript assignment must be a RegisterTensor."
                )

            # extract offsets and slice_dims
            indices = self.visit(stmt.target.slice)
            offsets = []
            slice_dims = []

            if not isinstance(indices, Sequence):
                indices = [indices]
            else:
                indices = list(indices)
            while len(indices) < len(dst_tensor.shape):
                indices.append(slice(None, None, None))
            for dim, index in enumerate(indices):
                if isinstance(index, slice):
                    if index.start is not None:
                        offsets.append(as_expr(index.start))
                    else:
                        offsets.append(Constant(0, data_type("int32")))
                    slice_dims.append(dim)
                else:
                    offsets.append(as_expr(index))

            # process rhs
            rhs = self.visit(stmt.value)
            if isinstance(stmt.value, RegisterTensor):
                rhs_tensor = rhs
            else:
                rhs_tensor = self.allocate_register(
                    dtype=dst_tensor.dtype, shape=[], f_init=lambda _: dst_tensor.dtype(rhs)
                )
            lhs_tensor = self.slice_register(
                tensor=dst_tensor,
                offsets=offsets,
                slice_dims=slice_dims,
                slice_shape=[dst_tensor.shape[dim] for dim in slice_dims],
            )

            if isinstance(stmt.op, ast.Add):
                result = self.add(x=lhs_tensor, y=rhs_tensor)
            elif isinstance(stmt.op, ast.Sub):
                result = self.sub(x=lhs_tensor, y=rhs_tensor)
            elif isinstance(stmt.op, ast.Mult):
                result = self.mul(x=lhs_tensor, y=rhs_tensor)
            elif isinstance(stmt.op, ast.Div):
                result = self.div(x=lhs_tensor, y=rhs_tensor)
            elif isinstance(stmt.op, ast.FloorDiv):
                result = self.div(x=lhs_tensor, y=rhs_tensor)
            elif isinstance(stmt.op, ast.Mod):
                result = self.mod(x=lhs_tensor, y=rhs_tensor)
            elif isinstance(stmt.op, ast.BitXor):
                result = self.bitwise_xor(x=lhs_tensor, y=rhs_tensor)
            else:
                raise TilusProgramError(self, stmt, "NotImplemented AugAssign for operator: {}".format(type(stmt.op)))

            self.slice_assign_register(output=dst_tensor, x=result, offsets=offsets, dims=slice_dims)
        else:
            raise TilusProgramError(
                self, stmt.target, "AugAssign only support variable name or RegisterTensor as target."
            )

    def visit_For(self, stmt: ast.For) -> None:
        # create loop vars
        iter_targets: list[ast.Name] = []
        if isinstance(stmt.target, (ast.List, ast.Tuple)):
            for target in stmt.target.elts:
                if not isinstance(target, ast.Name):
                    raise TilusProgramError(self, stmt, "For loop target must be a name.")
                iter_targets.append(target)
        else:
            if not isinstance(stmt.target, ast.Name):
                raise TilusProgramError(self, stmt, "For loop target must be a name.")
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
                raise TilusProgramError(
                    self, stmt, f"Expect {num_loop_vars} loop variables, but got {len(iter_targets)}."
                )

            with self.block(), self.scope() as for_scope:
                for var in loop_vars:
                    assert var.hint is not None
                    for_scope.bind(name=var.hint, var_or_value=var)
                for name, value in host_vars.items():
                    for_scope.bind(name, value)
                for s in stmt.body:
                    self.visit(s)
            body = self.pop_innermost_last()
            self.append(stmt_iter.generate_loop_statement(loop_vars=loop_vars, body=body))
        else:
            msg = "For loop iterable must be a one of the following types: \n1.\n  for ... in range(...): \n      ...\n"
            raise TilusProgramError(self, stmt.iter, msg)

    def visit_If(self, stmt: ast.If) -> None:
        cond = self.visit(stmt.test)

        if isinstance(cond, hidet_ir.Constant):
            cond = bool(cond)

        if isinstance(cond, bool):
            if cond:
                for s in stmt.body:
                    self.visit(s)
            else:
                for s in stmt.orelse:
                    self.visit(s)
        else:
            with self.if_then(cond=cond), self.scope():
                for s in stmt.body:
                    self.visit(s)
            if len(stmt.orelse) > 0:
                with self.otherwise(), self.scope():
                    for s in stmt.orelse:
                        self.visit(s)

    def visit_UnaryOp(
        self, expr: ast.UnaryOp
    ) -> Union[RegisterTensor, hidet_ir.Node, hidet_ir.BaseType, float, int, str]:
        if (
            isinstance(expr.op, ast.Invert)
            and isinstance(expr.operand, ast.Subscript)
            and isinstance(expr.operand.value, ast.Name)
        ):
            # handle the following syntax specially
            #  ~tensor[i, j, ...]
            # which gets the address of an element in global/shared tensor
            buf = self.visit(expr.operand.value)
            if isinstance(buf, (GlobalTensor, SharedTensor)):
                indices = self.visit(expr.operand.slice)
                if not isinstance(indices, Sequence):
                    indices = [indices]
                if len(indices) != len(buf.shape):
                    raise TilusProgramError(
                        self,
                        expr.operand,
                        "Index dimension {} does not match tensor shape {}.".format(len(indices), buf.shape),
                    )
                return self.tensor_item_ptr(buf, space="generic")
            elif isinstance(buf, RegisterTensor):
                raise ValueError("Can not addressing the element of a RegisterTensor.")

        value = self.visit(expr.operand)
        if isinstance(value, RegisterTensor):
            if isinstance(expr.op, ast.UAdd):
                # +v
                return value
            elif isinstance(expr.op, ast.USub):
                # -v
                return self.neg(value)
            elif isinstance(expr.op, ast.Not):
                # not v
                return self.logical_not(value)
            else:
                raise TilusProgramError(self, expr, "Can not recognize unary operator for RegisterTensor.")
        elif isinstance(value, hidet_ir.Node):
            if isinstance(expr.op, ast.Not):
                # not v
                assert isinstance(value, hidet_ir.Expr)
                return hidet_ir.logical_not(value)
            elif isinstance(expr.op, ast.Invert):
                # there are two cases for a ~ operator: ~something
                # case 1: get the address of an expression
                # case 2: get the pointer type that points to the given type
                from hidet.ir.expr import Address
                from hidet.ir.type import BaseType

                if isinstance(value, BaseType):
                    return ~value
                else:
                    assert isinstance(value, hidet_ir.Expr)
                    return Address(value)
            elif isinstance(expr.op, ast.UAdd):
                # +v
                return value
            elif isinstance(expr.op, ast.USub):
                # -v
                assert isinstance(value, hidet_ir.Expr)
                return -value
            else:
                raise TilusProgramError(self, expr, "Can not recognize unary operator.")
        else:
            op_dict: dict[Any, Callable] = {
                ast.UAdd: operator.pos,
                ast.USub: operator.neg,
                ast.Not: operator.not_,
            }
            return op_dict[type(expr.op)](value)

    def visit_While(self, stmt: ast.While) -> None:
        cond = self.visit(stmt.test)
        with self.while_loop(cond=as_expr(cond)), self.scope():
            for s in stmt.body:
                self.visit(s)

    def visit_Break(self, stmt: ast.Break) -> None:
        self.brk()

    def visit_Return(self, stmt: ast.Return) -> None:
        if stmt.value is not None:
            raise TilusProgramError(self, stmt, "Return statement in Tilus Script does not support returning a value.")
        self.ret()

    def visit_Slice(self, expr: ast.Slice) -> slice:
        return slice(
            self.visit(expr.lower) if expr.lower is not None else None,
            self.visit(expr.upper) if expr.upper is not None else None,
            self.visit(expr.step) if expr.step is not None else None,
        )

    def visit_With(self, stmt: ast.With) -> None:
        with_items = stmt.items
        if len(with_items) != 1:
            raise TilusProgramError(self, stmt, "Tilus currently do not support multiple with items.")

        with_item: ast.withitem = with_items[0]
        with_item_visited = self.visit(with_item.context_expr)

        if not isinstance(with_item_visited, TilusContext):
            raise TilusProgramError(
                self,
                with_item.context_expr,
                "The context manager in Tilus Script must be a tilus.lang.constructs.contexts.TilusContext object.",
            )

        with_context: TilusContext = with_item_visited

        with self.block(), self.scope():
            # bind the value to the context variable
            if with_item.optional_vars is not None:
                if not isinstance(with_item.optional_vars, ast.Name):
                    raise TilusProgramError(
                        self, with_item.optional_vars, "Tilus only support binding to a single name."
                    )
                bind_value = with_context.bind_value()
                if bind_value is None:
                    raise TilusProgramError(self, with_item.optional_vars, "The context does not have a bind value.")
                bind_name = with_item.optional_vars.id

                if isinstance(bind_value, hidet_ir.Expr):
                    from hidet.ir.tools import infer_type

                    bind_var = self.declare(type=infer_type(bind_value), init=bind_value, hint=bind_name)
                    self.bind(bind_name, var_or_value=bind_var)
                else:
                    self.bind(with_item.optional_vars.id, bind_value)

            for body_stmt in stmt.body:
                self.visit(body_stmt)

        with_body = self.pop_innermost_last()
        processed_body = with_context.post_process(with_body)
        assert isinstance(processed_body, Stmt)
        self.append(processed_body)

    def process_generator(self, elt: ast.expr, generators: list[ast.comprehension]) -> list:
        if len(generators) == 0:
            return [self.visit(elt)]
        else:
            generator = generators[0]
            if generator.is_async:
                raise TilusProgramError(self, None, "Hidet currently do not support async generator.")
            assert isinstance(generator, ast.comprehension)
            iterator = self.visit(generator.iter)
            names: list[str] = []
            if isinstance(generator.target, ast.Name):
                names = [generator.target.id]
            elif isinstance(generator.target, ast.Tuple):
                for target in generator.target.elts:
                    if not isinstance(target, ast.Name):
                        raise TilusProgramError(
                            self, target, "Hidet currently only support binding a single name or a tuple of names"
                        )
                    names.append(target.id)
            else:
                raise TilusProgramError(
                    self,
                    generator.target,
                    "Hidet do not support generator target with type {}.".format(type(generator.target)),
                )
            result = []
            for it in iterator:
                if len(names) == 1:
                    self.bind(names[0], it)
                else:
                    if len(names) != len(it):
                        raise TilusProgramError(
                            self, None, "Can not unpack {} values to {} names.".format(len(it), len(names))
                        )
                    for name, value in zip(names, it):
                        self.bind(name, value)
                if not all(self.visit(cond) for cond in generator.ifs):
                    continue
                result.extend(self.process_generator(elt, generators[1:]))
            return result

    def visit_ListComp(self, expr: ast.ListComp) -> list:
        return self.process_generator(expr.elt, expr.generators)

    def visit_DictComp(self, expr: ast.DictComp) -> dict:
        kv_pair = ast.Tuple([expr.key, expr.value])
        kv_pairs = self.process_generator(kv_pair, expr.generators)
        return {k: v for k, v in kv_pairs}

    def visit_SetComp(self, expr: ast.SetComp) -> set:
        values = self.process_generator(expr.elt, expr.generators)
        return set(values)

    def visit_GeneratorExp(self, expr: ast.GeneratorExp) -> list:
        return self.process_generator(expr.elt, expr.generators)

    def visit_Pass(self, stmt: ast.Pass) -> None:
        self.append(SeqStmt(tuple()))
