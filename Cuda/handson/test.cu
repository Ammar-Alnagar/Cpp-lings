#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d!\n", idx);
}

int main() {
    printf("Hello from CPU!\n");
    
    // Launch kernel with 1 block of 8 threads
    hello_from_gpu<<<1, 8>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("Program completed successfully!\n");
    return 0;
}