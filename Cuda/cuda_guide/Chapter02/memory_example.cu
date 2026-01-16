#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Kernel demonstrating different memory types
__global__ void memoryTypesExample(float *global_mem, float *result, int n) {
    // Shared memory - fast memory shared among threads in a block
    __shared__ float shared_mem[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load data to shared memory
    if (tid < n && local_tid < 256) {
        shared_mem[local_tid] = global_mem[tid];
    }
    __syncthreads();
    
    // Perform computation using shared memory
    if (tid < n) {
        // Access global memory
        float val = global_mem[tid];
        
        // Use shared memory for faster access
        if (local_tid < 256) {
            val += shared_mem[local_tid];
        }
        
        // Store result
        result[tid] = val * 2.0f;
    }
}

// Kernel demonstrating coalesced vs strided access
__global__ void coalescedAccess(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced access: consecutive threads access consecutive memory
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel demonstrating unified memory
__global__ void unifiedMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main() {
    // Allocate host memory
    float *h_data = (float*)malloc(N * sizeof(float));
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_data, *d_result;
    gpuErrchk(cudaMalloc(&d_data, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_result, N * sizeof(float)));
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Setup execution configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch kernel demonstrating different memory types
    memoryTypesExample<<<gridSize, blockSize>>>(d_data, d_result, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Demonstrate unified memory
    float *unified_data;
    gpuErrchk(cudaMallocManaged(&unified_data, N * sizeof(float)));
    
    // Initialize unified memory on CPU
    for (int i = 0; i < N; i++) {
        unified_data[i] = i * 1.0f;
    }
    
    // Process with GPU
    unifiedMemoryKernel<<<gridSize, blockSize>>>(unified_data, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Access unified memory result on CPU
    printf("First 10 values after GPU processing:\n");
    for (int i = 0; i < 10; i++) {
        printf("unified_data[%d] = %.2f\n", i, unified_data[i]);
    }
    
    // Cleanup
    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_result));
    gpuErrchk(cudaFree(unified_data));
    free(h_data);
    
    printf("Memory management example completed successfully.\n");
    
    return 0;
}