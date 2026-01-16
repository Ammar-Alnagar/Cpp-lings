#include <stdio.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// Example kernel for debugging
__global__ void debugKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Add bounds checking to prevent out-of-bounds access
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Example kernel with potential issues for debugging practice
__global__ void problematicKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This could cause out-of-bounds access if not careful
    if (idx < n) {
        // Potential issue: if idx+1 >= n, accessing invalid memory
        if (idx + 1 < n) {
            data[idx] = data[idx] + data[idx + 1];
        }
    }
}

// Kernel for profiling example
__global__ void profiledKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Perform some computation to measure
        float val = input[idx];
        val = val * val + sqrtf(val) + expf(-val);
        output[idx] = val;
    }
}

// Memory debugging example
__global__ void memoryCheckKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Initialize memory
        data[idx] = idx * 1.0f;
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
    const int N = 1024;
    const size_t size = N * sizeof(float);
    
    printf("=== CUDA Debugging and Profiling Example ===\n");
    
    // Allocate host and device memory
    float *h_data = (float*)malloc(size);
    float *d_data1, *d_data2, *d_temp;
    
    gpuErrchk(cudaMalloc(&d_data1, size));
    gpuErrchk(cudaMalloc(&d_data2, size));
    gpuErrchk(cudaMalloc(&d_temp, size));
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.1f;
    }
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_data1, h_data, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_data2, h_data, size, cudaMemcpyHostToDevice));
    
    printf("Memory allocated and initialized successfully.\n");
    
    // NVTX profiling markers example
    nvtxRangePush("Safe Kernel Execution");
    
    // Setup execution configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch debug kernel with error checking
    debugKernel<<<gridSize, blockSize>>>(d_data1, N);
    gpuErrchk(cudaGetLastError());  // Check for launch errors
    gpuErrchk(cudaDeviceSynchronize());  // Check for execution errors
    
    nvtxRangePop(); // Safe Kernel Execution
    
    printf("Debug kernel executed successfully.\n");
    
    // Launch problematic kernel (fixed version)
    nvtxRangePush("Problematic Kernel Execution");
    
    problematicKernel<<<gridSize, blockSize>>>(d_data2, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    nvtxRangePop(); // Problematic Kernel Execution
    
    printf("Problematic kernel executed successfully.\n");
    
    // Profiling example
    nvtxRangePush("Profiling Example");
    
    // Copy original data back to d_data1 for profiling test
    gpuErrchk(cudaMemcpy(d_data1, h_data, size, cudaMemcpyHostToDevice));
    
    // Launch profiling kernel
    profiledKernel<<<gridSize, blockSize>>>(d_data1, d_temp, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    nvtxRangePop(); // Profiling Example
    
    printf("Profiled kernel executed successfully.\n");
    
    // Memory debugging example
    nvtxRangePush("Memory Debugging");
    
    memoryCheckKernel<<<gridSize, blockSize>>>(d_data1, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host to verify
    gpuErrchk(cudaMemcpy(h_data, d_data1, size, cudaMemcpyDeviceToHost));
    
    printf("First 10 values after memory check kernel:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");
    
    nvtxRangePop(); // Memory Debugging
    
    // Demonstrate unified memory debugging
    printf("\n=== Unified Memory Example ===\n");
    
    float *unified_data;
    gpuErrchk(cudaMallocManaged(&unified_data, size));
    
    // Initialize on CPU
    for (int i = 0; i < N; i++) {
        unified_data[i] = i * 0.5f;
    }
    
    // Process on GPU
    debugKernel<<<gridSize, blockSize>>>(unified_data, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Access result on CPU
    printf("First 10 values after unified memory processing:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", unified_data[i]);
    }
    printf("\n");
    
    // Cleanup unified memory
    gpuErrchk(cudaFree(unified_data));
    
    // Test error handling with invalid parameters
    printf("\n=== Error Handling Test ===\n");
    
    // This would normally cause an error, but we're not executing it
    // cudaMalloc(&d_data1, 0);  // Would cause an error
    
    printf("Error handling test completed.\n");
    
    // Cleanup
    gpuErrchk(cudaFree(d_data1));
    gpuErrchk(cudaFree(d_data2));
    gpuErrchk(cudaFree(d_temp));
    free(h_data);
    
    printf("\nAll debugging and profiling examples completed successfully.\n");
    printf("Run with cuda-gdb for interactive debugging:\n");
    printf("  cuda-gdb ./debug_profiling_example\n");
    printf("Run with cuda-memcheck for memory error detection:\n");
    printf("  cuda-memcheck ./debug_profiling_example\n");
    printf("Run with Nsight Compute for detailed profiling:\n");
    printf("  ncu --set full ./debug_profiling_example\n");
    
    return 0;
}