// Direct CUDA test of stmatrix hardware behavior.
// Each thread stores a known pattern and we observe where it lands in shared memory.
// This tells us the exact thread-to-position mapping.

#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

__device__ uint32_t cvta_to_shared(void* ptr) {
    uint32_t ret;
    asm("{.reg.u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr;}"
        : "=r"(ret) : "l"(ptr));
    return ret;
}

__global__ void test_stmatrix_kernel(half* output) {
    __shared__ half smem[8 * 8];  // 8x8 matrix in shared memory

    int tid = threadIdx.x;  // 0-31

    // Zero shared memory
    if (tid < 32) {
        for (int i = tid; i < 64; i += 32)
            smem[i] = __float2half(0.0f);
    }
    __syncthreads();

    // Each thread's register: encode (thread_id * 100 + element_index)
    // so we can trace where each thread's data ends up
    // u32 packs 2 fp16: low half = tid*100+0, high half = tid*100+1
    __half lo = __float2half((float)(tid * 100));
    __half hi = __float2half((float)(tid * 100 + 1));
    uint32_t reg;
    asm("mov.b32 %0, {%1, %2};" : "=r"(reg) : "h"(*reinterpret_cast<unsigned short*>(&lo)), "h"(*reinterpret_cast<unsigned short*>(&hi)));

    // All threads provide the SAME address: start of shared memory row 0
    // This way we can see which thread writes where
    uint32_t addr = cvta_to_shared(&smem[0]);

    // stmatrix.sync.aligned.m8n8.x1.shared.b16
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                 : : "r"(addr), "r"(reg));
    __syncthreads();

    // Read out shared memory
    if (tid == 0) {
        printf("stmatrix x1 with ALL threads pointing to row 0:\n");
        for (int r = 0; r < 8; r++) {
            printf("  smem row %d:", r);
            for (int c = 0; c < 8; c++) {
                printf(" %6.0f", __half2float(smem[r * 8 + c]));
            }
            printf("\n");
        }
    }
    __syncthreads();

    // Now test with proper per-thread addresses (each group of 4 points to different row? or each thread to different row?)
    // Zero shared
    for (int i = tid; i < 64; i += 32)
        smem[i] = __float2half(0.0f);
    __syncthreads();

    // Test 1: each thread points to row tid/4 (spatial(8,4) hypothesis)
    {
        int row = tid / 4;
        uint32_t addr2 = cvta_to_shared(&smem[row * 8]);
        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                     : : "r"(addr2), "r"(reg));
        __syncthreads();
        if (tid == 0) {
            printf("\nstmatrix x1 with addr = row[tid/4] (spatial(8,4) hypothesis):\n");
            for (int r = 0; r < 8; r++) {
                printf("  smem row %d:", r);
                for (int c = 0; c < 8; c++) {
                    printf(" %6.0f", __half2float(smem[r * 8 + c]));
                }
                printf("\n");
            }
        }
        __syncthreads();
    }

    // Zero shared
    for (int i = tid; i < 64; i += 32)
        smem[i] = __float2half(0.0f);
    __syncthreads();

    // Test 2: each thread points to row tid%8 (column_spatial(8,4) hypothesis)
    {
        int row = tid % 8;
        uint32_t addr3 = cvta_to_shared(&smem[row * 8]);
        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                     : : "r"(addr3), "r"(reg));
        __syncthreads();
        if (tid == 0) {
            printf("\nstmatrix x1 with addr = row[tid%%8] (column_spatial hypothesis):\n");
            for (int r = 0; r < 8; r++) {
                printf("  smem row %d:", r);
                for (int c = 0; c < 8; c++) {
                    printf(" %6.0f", __half2float(smem[r * 8 + c]));
                }
                printf("\n");
            }
        }
    }
}

int main() {
    half* d_out;
    cudaMalloc(&d_out, 64 * sizeof(half));
    test_stmatrix_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
