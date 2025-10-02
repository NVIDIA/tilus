#include <tvm/ffi/type_traits.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace tvm {
namespace ffi {

// Template specialization for half* (CUDA half precision)
template <>
struct TypeTraits<half*> : public FallbackOnlyTraitsBase<half*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "float16*"; }  

  TVM_FFI_INLINE static half* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLFloat || src->dtype.bits != 16) {
      TVM_FFI_THROW(ValueError) << "DLTensor dtype is not 16 bit float";
    }
    return reinterpret_cast<half*>(src->data);
  }
};

template <>
struct TypeTraits<__nv_bfloat16*> : public FallbackOnlyTraitsBase<__nv_bfloat16*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "bfloat16*"; }  

  TVM_FFI_INLINE static __nv_bfloat16* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLBfloat || src->dtype.bits != 16) {
      TVM_FFI_THROW(ValueError) << "DLTensor dtype is not bfloat16";
    }
    return reinterpret_cast<__nv_bfloat16*>(src->data);
  }
};

// Template specialization for float*
template <typename Float>
struct TypeTraits<Float*, std::enable_if_t<std::is_floating_point_v<Float>>> : public FallbackOnlyTraitsBase<Float*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "float" + std::string(std::numeric_limits<Float>::digits); }  

  TVM_FFI_INLINE static Float* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLFloat || src->dtype.bits != std::numeric_limits<Float>::digits) {
      TVM_FFI_THROW(ValueError) << "DLTensor dtype is not " << std::numeric_limits<Float>::digits << " bits";
    }
    return reinterpret_cast<Float*>(src->data);
  }
};

// Template specialization for integral types
template <typename Int>
struct TypeTraits<Int*, std::enable_if_t<std::is_integral_v<Int>>> : public FallbackOnlyTraitsBase<Int*, DLTensor*> {
  TVM_FFI_INLINE static std::string TypeStr() { return "int" + std::string(std::numeric_limits<Int>::digits); }  

  TVM_FFI_INLINE static Int* ConvertFallbackValue(DLTensor* src) {
    if (src->dtype.code != kDLInt || src->dtype.bits != std::numeric_limits<Int>::digits) {
      TVM_FFI_THROW(ValueError) << "DLTensor dtype is not " << std::numeric_limits<Int>::digits << " bits";
    }
    return reinterpret_cast<Int*>(src->data);
  }
};

} // namespace ffi
} // namespace tvm
