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
import os
from typing import Sequence, Union

from hidet.backend.codegen import Codegen, CPUCodegen, CUDACodegen, HIPCodegen
from hidet.ir.expr import Constant, Expr, Var
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.target import Target
from hidet.ir.type import (
    ArrayType,
    DataType,
    FuncType,
    OpaqueType,
    PointerType,
    ReferenceType,
    TensorPointerType,
    TensorType,
    VoidType,
)
from hidet.ir.utils.call_graph import CallGraph
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.utils.py import prod

from tilus.extensions.hidet.ir.dtypes.vector import uint32x1, uint32x2, uint32x4


class UpdatedCUDACodeGen(CUDACodegen):
    def param_declare_v2(self, func_kind: str, v: Var) -> Doc:
        v_type = v.type
        name_doc = self(v)
        restrict_item = " " if "internal" in func_kind else " __restrict__ "
        if isinstance(v_type, DataType):
            dtype_doc = self(v_type)
            return dtype_doc + " " + name_doc
        elif isinstance(v_type, PointerType):
            if len(v_type.specifiers) > 0:
                attr_doc = doc_join([self(attr) for attr in v_type.specifiers], sep=" ") + " "
            else:
                attr_doc = Doc()
            if isinstance(v_type.base_type, OpaqueType) and v_type.base_type.cpp_name == "CUtensorMap":
                attr_doc = "const " + attr_doc  # todo: make it more general
            dtype = v_type.base_type
            base_type_doc = self(dtype)
            if v_type.use_bracket:
                return attr_doc + base_type_doc + " " + name_doc + "[]"
            elif func_kind == "public" and isinstance(dtype, VoidType):
                return attr_doc + "void_p " + name_doc
            else:
                return attr_doc + base_type_doc + " *" + restrict_item + name_doc
        elif isinstance(v_type, TensorPointerType):
            dtype = v_type.tensor_type.dtype
            base_type_doc = self(dtype)
            return base_type_doc + " *" + restrict_item + name_doc
        elif isinstance(v_type, ReferenceType):
            if isinstance(v_type.base_type, DataType):
                base_type_doc = self(v_type.base_type)
                return base_type_doc + " &" + name_doc
            else:
                raise NotImplementedError()
        elif isinstance(v_type, TensorType):
            dtype = v_type.dtype
            base_type_doc = self(dtype)
            return base_type_doc + " *" + restrict_item + name_doc
        elif isinstance(v_type, ArrayType):
            base_type_doc = self(v_type.base_type)
            return base_type_doc + " " + name_doc + "[" + self(v_type.size) + "]"
        elif isinstance(v_type, OpaqueType):
            dtype_doc = self(v_type)
            if v_type.cpp_name == "CUtensorMap":
                dtype_doc = "const __grid_constant__ " + dtype_doc
            return dtype_doc + " " + name_doc
        else:
            raise ValueError()

    def visit_Function(self, func: Function) -> Doc:
        self.namer.clear()

        doc = NewLine()

        # ret
        if func.kind == "cuda_kernel":
            doc += "static __global__ "
        elif func.kind == "cuda_internal":
            doc += "static __device__ __forceinline__ "
        elif func.kind == "cpu_kernel":
            doc += "static "
        elif func.kind == "cpu_internal":
            doc += "static __forceinline__ "
        elif func.kind == "public":
            if self.ir_module.namespace == "":
                doc += "DLL "
        else:
            raise ValueError(f"Unknown function kind: {func.kind}")

        doc += self(func.ret_type)

        if "cuda.cluster_dim" in func.attrs:
            from hidet.transforms.generate_launch_func import _normalize_dim3

            cluster_dims = _normalize_dim3(func.attrs["cuda.cluster_dim"])  # type: ignore
            doc += f" __cluster_dims__({cluster_dims[0]}, {cluster_dims[1]}, {cluster_dims[2]})"

            self.require_cooperative_groups = True

        # launch bound for grid worker
        if func.kind == "cuda_kernel":
            block_dim = func.attrs["cuda.block_dim"]
            if isinstance(block_dim, list):
                block_dim = prod(block_dim)
            if isinstance(block_dim, (Constant, int)):
                if "cuda.min_blocks" in func.attrs:
                    min_blocks = func.attrs["cuda.min_blocks"]
                    if isinstance(min_blocks, (Constant, int)):
                        doc += f" __launch_bounds__({block_dim}, {min_blocks})"
                    else:
                        doc += f" __launch_bounds__({block_dim})"
                else:
                    doc += f" __launch_bounds__({block_dim})"

        # func name
        canonized_func_name = self.canonize_funcname(func.name)
        doc += " " + canonized_func_name

        # parameters
        doc += "("
        param_docs = []
        for param in func.params:
            param_docs.append(self.param_declare_v2(func.kind, param))
        doc += doc_join(param_docs, Text(", "))
        doc += ") {"

        # body
        body_doc = self(func.body)
        if func.kind == "public" and self.ir_module.namespace == "":
            body_doc = self.wrap_try_catch(body_doc, func.ret_type)
        doc += body_doc.indent()

        doc += NewLine() + "}"

        return doc

    def require_headers(self) -> Doc:
        doc = Text("#include <tvm/ffi/function.h>") + NewLine()
        doc += Text("#include <hidet/tvm/ffi/extra_type_traits.h>") + NewLine()
        doc += Text("#include <hidet/void_p.h>") + NewLine()
        doc += super().require_headers()
        return doc

    def visit_IRModule(self, module: IRModule) -> Doc:
        if module.namespace != "":
            raise NotImplementedError("Namespace is not supported")

        self.ir_module = module
        doc = Doc()

        for name, func_var in module.extern_functions.items():
            assert isinstance(func_var.type, FuncType)
            doc += self.declare_function(name, func_var.type) + NewLine()

        # define global variables
        for name, var in module.global_vars.items():
            if name in module.functions:
                continue
            doc += self.local_var_declare(var) + ";" + NewLine()

        # define functions
        call_graph = CallGraph(module)
        for node in call_graph.reversed_order:
            doc += self(node.func) + NewLine()

        # define tvm-ffi registries
        for name, func in module.functions.items():
            if func.kind != "public":
                continue
            doc += Text(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({name}, hidet_{name});") + NewLine()

        doc = self.require_headers() + doc
        return doc

    def scalar_literal(self, value: Expr, dtype: DataType) -> Doc:
        ret: Union[str, Doc]
        if dtype == uint32x1:
            ret = "make_uint1({})".format(int(value[0]))
        elif dtype == uint32x2:
            ret = "make_uint2({}, {})".format(int(value[0]), int(value[1]))
        elif dtype == uint32x4:
            ret = "make_uint4({}, {}, {}, {})".format(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
        else:
            ret = super().scalar_literal(value, dtype)
        if isinstance(ret, str):
            ret = Text(ret)
        return ret


def codegen(ir_module: Union[IRModule, Sequence[IRModule]], src_out_path: str, target: Union[str, Target]) -> str:
    if isinstance(target, str):
        target = Target.from_string(target)

    gen: Codegen
    if target.name == "cuda":
        gen = UpdatedCUDACodeGen()
    elif target.name == "hip":
        gen = HIPCodegen()
    elif target.name == "cpu":
        gen = CPUCodegen()
    else:
        raise ValueError(f"Unknown target: {target}")

    code = ""
    if isinstance(ir_module, Sequence):
        for m in ir_module:
            doc = gen(m)
            code += str(doc) + "\n"
    else:
        doc = gen(ir_module)
        code = str(doc)
    if src_out_path is not None:
        dir_path = os.path.dirname(src_out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(src_out_path, "w") as f:
            f.write(code)
    return code
