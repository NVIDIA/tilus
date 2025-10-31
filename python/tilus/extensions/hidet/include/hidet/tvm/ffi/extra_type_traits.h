#pragma once

#include <tvm/ffi/type_traits.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <hidet/void_p.h>

namespace tvm {
namespace ffi {


inline std::string dtype_to_str(DLDataType dtype) {
  switch (dtype.code) {
    case kDLInt: return "int" + std::to_string(dtype.bits);
    case kDLUInt: return "uint" + std::to_string(dtype.bits);
    case kDLFloat: return "float" + std::to_string(dtype.bits);
    case kDLBfloat: return "bfloat16";
    case kDLOpaqueHandle: return "opaque_handle";
    case kDLComplex: return "complex" + std::to_string(dtype.bits);
    case kDLBool: return "bool";
    default: return "dtype(code=" + std::to_string(dtype.code) + ", bits=" + std::to_string(dtype.bits) + ", lanes=" + std::to_string(dtype.lanes) + ")";
  }
}


template <>
struct TypeTraits<void_p> : public FallbackOnlyTraitsBase<void_p, DLTensor*, int64_t, uint64_t> {
  TVM_FFI_INLINE static std::string TypeStr() { return "void_p"; }

  TVM_FFI_INLINE static void_p ConvertFallbackValue(DLTensor* src) {
    return src->data;
  }

  TVM_FFI_INLINE static void_p ConvertFallbackValue(int64_t src) {
    return reinterpret_cast<void*>(src);
  }

  TVM_FFI_INLINE static void_p ConvertFallbackValue(uint64_t src) {
    return reinterpret_cast<void*>(src);
  }
};

// Template specialization for half*
template <>
struct TypeTraits<half*> : public FallbackOnlyTraitsBase<half*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "float16*"; }

  TVM_FFI_INLINE static half* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLFloat || src->dtype.bits != 16) {
      TVM_FFI_THROW(ValueError) << "Expect a tensor with 16 bit float16, got a tensor with dtype " << dtype_to_str(src->dtype);
    }
    return reinterpret_cast<half*>(src->data);
  }
};

// Template specialization for __nv_bfloat16*
template <>
struct TypeTraits<__nv_bfloat16*> : public FallbackOnlyTraitsBase<__nv_bfloat16*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "bfloat16*"; }

  TVM_FFI_INLINE static __nv_bfloat16* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLBfloat || src->dtype.bits != 16) {
      TVM_FFI_THROW(ValueError) << "Expect a tensor with 16 bit bfloat16, got a tensor with dtype " << dtype_to_str(src->dtype);
    }
    return reinterpret_cast<__nv_bfloat16*>(src->data);
  }
};

// Template specialization for float*, double*
template <typename Float>
struct TypeTraits<Float*, std::enable_if_t<std::is_floating_point_v<Float>>> : public FallbackOnlyTraitsBase<Float*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "float" + std::to_string(sizeof(Float) * 8); }

  TVM_FFI_INLINE static Float* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLFloat || src->dtype.bits != sizeof(Float) * 8) {
      TVM_FFI_THROW(ValueError) << "Expect a tensor with " << sizeof(Float) * 8 << " bit floating-point, got a tensor with dtype " << dtype_to_str(src->dtype);
    }
    return reinterpret_cast<Float*>(src->data);
  }
};

// Template specialization for int32_t*, int16_t*, etc.
template <typename Int>
struct TypeTraits<Int*, std::enable_if_t<std::is_signed_v<Int> && std::is_integral_v<Int>>> : public FallbackOnlyTraitsBase<Int*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "int" + std::to_string(sizeof(Int) * 8); }

  TVM_FFI_INLINE static Int* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLInt || src->dtype.bits != sizeof(Int) * 8) {
      TVM_FFI_THROW(ValueError) << "Expect a tensor with " << sizeof(Int) * 8 << " bit signed integer, got a tensor with dtype " << dtype_to_str(src->dtype);
    }
    return reinterpret_cast<Int*>(src->data);
  }
};

// Template specialization for uint32_t*, uint16_t*, etc.
template<typename UInt>
struct TypeTraits<UInt*, std::enable_if_t<std::is_unsigned_v<UInt> && std::is_integral_v<UInt>>> : public FallbackOnlyTraitsBase<UInt*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "uint" + std::to_string(sizeof(UInt) * 8); }

  TVM_FFI_INLINE static UInt* ConvertFallbackValue(DLTensor* src) {
    if ((src->dtype.code != kDLUInt || src->dtype.bits != sizeof(UInt) * 8)
        && (src->dtype.code != kDLBool || src->dtype.bits != 8)
      ) {
      TVM_FFI_THROW(ValueError) << "Expect a tensor with " << sizeof(UInt) * 8 << " bit unsigned integer, got a tensor with dtype " << dtype_to_str(src->dtype);
    }
    return reinterpret_cast<UInt*>(src->data);
  }
};

} // namespace ffi
} // namespace tvm
