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
# from hidet.ir.type import DataType
# from hidet.ir.expr import Cast, Constant, reinterpret, convert, cast
# from hidet.ir.func import Function
# from hidet.ir.dtypes import float32, float8_e5m2, uint8
# from hidet.ir.functors import IRRewriter
# from hidet.ir.tools import TypeInfer
# from hidet.transforms.base import FunctionPass
# from mutis.extension.primitives.gpgpu.cast import cast_subbyte_float_to_f32, cast_subbyte_float_from_f32
# from mutis.extension.primitives.gpgpu.reinterpret import reinterpret
#
#
# class LowerFloat8CastRewriter(IRRewriter):
#     def __init__(self):
#         super().__init__()
#         self.type_infer = TypeInfer()
#
#     def visit_Constant(self, e: Constant):
#         if e.type in [float8_e5m2]:
#             return self.visit(reinterpret(cast_subbyte_float_from_f32(convert(e.value), e.type), e.type))
#         return super().visit_Constant(e)
#
#     def visit_Cast(self, e: Cast):
#         src_type = self.type_infer(e.expr)
#         dst_type = e.target_type
#
#         if isinstance(src_type, DataType) and isinstance(dst_type, DataType):
#             if src_type in [float8_e5m2] or dst_type in [float8_e5m2]:
#                 if dst_type == float32:
#                     return cast_subbyte_float_to_f32(reinterpret(self.visit(e.expr), uint8), src_type)
#                 elif src_type == float32:
#                     return reinterpret(cast_subbyte_float_from_f32(self.visit(e.expr), dst_type), dst_type)
#                 else:
#                     return self.visit(cast(cast(self.visit(e.expr), float32), dst_type))
#
#         return super().visit_Cast(e)
#
#
# class LowerFloat8CastPass(FunctionPass):
#     def process_func(self, func: Function) -> Function:
#         rewriter = LowerFloat8CastRewriter()
#         return rewriter(func)
#
#
# def lower_float8_cast_pass():
#     return LowerFloat8CastPass()
