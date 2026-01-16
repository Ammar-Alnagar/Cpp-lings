// Define macros to handle compatibility issues before including headers
#define _GLIBCXX_USE_C99_MATH 0
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function that runs on the GPU
__global__ void helloFromGPU(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("Hello from GPU thread %d!\n", idx);
    }
}

int main() {
    int numThreads = 8;
    
    // Define grid and block dimensions
    int blockSize = 4;
    int gridSize = (numThreads + blockSize - 1) / blockSize;
    
    printf("Hello from CPU!\n");
    
    // Launch kernel with specified grid and block dimensions
    helloFromGPU<<<gridSize, blockSize>>>(numThreads);
    
    // Wait for GPU to finish before CPU continues
    cudaDeviceSynchronize();
    
    return 0;
}