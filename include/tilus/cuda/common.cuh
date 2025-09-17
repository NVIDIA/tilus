#include <cuda.h>
#include <cuda_runtime_api.h>


inline void __check_cu_error(CUresult code, const char* op, const char* file, int line) {
    if (code != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(code, &error_name);
        cuGetErrorString(code, &error_string);
        std::cerr << "CUDA Driver API call failed: " << op << " at " << file << ":" << line << "\n";
        std::cerr << "  Error: " << error_name << " (" << error_string << ")\n";
        exit(1);
    }
}


inline void __check_cuda_error(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Runtime API failed: " << op << " at " << file << ":" << line << "\n";
        std::cerr << "  Error: " << cudaGetErrorString(code) << "\n";
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(op) __check_cuda_error((op), #op, __FILE__, __LINE__)
#define CHECK_CU_ERROR(op) __check_cu_error((op), #op, __FILE__, __LINE__)
