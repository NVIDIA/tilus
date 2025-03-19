import os
from typing import List, Sequence, Tuple, Union

from hidet.backend.codegen import Codegen, CPUCodegen, HIPCodegen
from hidet.backend.codegen import CUDACodegen as OriginalCUDACodegen
from hidet.ir import Add
from hidet.ir.module import IRModule
from hidet.ir.target import Target
from hidet.ir.type import DataType
from hidet.utils.doc import Doc, Text
from tilus.extensions.hidet.utils.doc import doc_join, doc_strip_parentheses


class CUDACodegen(OriginalCUDACodegen):
    def visit_List(self, lst: List) -> Doc:
        return doc_join([doc_strip_parentheses(self(v)) for v in lst], ", ")

    def visit_Tuple(self, tp: Tuple) -> Doc:
        return doc_join([doc_strip_parentheses(self(v)) for v in tp], ", ")

    def visit_DataType(self, t: DataType) -> Doc:
        scalar_type_map = {
            "uint16x1": "ushort1",
            "uint16x2": "ushort2",
            "uint16x4": "ushort4",
            "uint32x1": "uint1",
            "uint32x2": "uint2",
            "uint32x4": "uint4",
            "uint64x1": "ulonglong1",
            "uint64x2": "ulonglong2",
            "uint64x4": "ulonglong4",
        }
        if t.name in scalar_type_map:
            return Text(scalar_type_map[t.name])
        else:
            return super().visit_DataType(t)

    def visit_Add(self, e: Add) -> Doc:
        addition_chain = []

        def expand(ee):
            if isinstance(ee.a, Add):
                expand(ee.a)
                addition_chain.append(ee.b)
            else:
                addition_chain.append(ee.a)
                addition_chain.append(ee.b)

        expand(e)

        assert len(addition_chain) >= 2

        doc = Doc()
        doc += "("
        for i, item in enumerate(addition_chain):
            doc += self(item)
            if i != len(addition_chain) - 1:
                doc += " + "
        doc += ")"
        c_type = self.type_infer(e)
        if isinstance(c_type, DataType) and c_type.is_integer() and c_type.nbytes < 4:
            doc = "(" + self(c_type) + ")" + doc
        return doc


def codegen(ir_module: Union[IRModule, Sequence[IRModule]], src_out_path: str, target: Union[str, Target]) -> str:
    if isinstance(target, str):
        target = Target.from_string(target)

    gen: Codegen
    if target.name == "cuda":
        gen = CUDACodegen()
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
