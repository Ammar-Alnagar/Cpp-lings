# Chapter 5: Debugging and Profiling CUDA Applications

## Overview
Debugging and profiling are essential skills for developing high-performance CUDA applications. This chapter covers various tools and techniques for identifying bugs, measuring performance, and optimizing CUDA code.

## Debugging CUDA Applications

### Common CUDA Errors
CUDA applications can fail due to various reasons. Understanding common errors helps in debugging:

1. **Memory access violations** - Accessing out-of-bounds memory
2. **Kernel launch failures** - Invalid grid/block dimensions
3. **Race conditions** - Improper synchronization
4. **Resource exhaustion** - Exceeding register/shared memory limits

### Runtime Error Checking
Always check for CUDA runtime errors:

```cuda
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void safeCudaCallExample() {
    float *d_data;
    gpuErrchk(cudaMalloc(&d_data, 1024 * sizeof(float)));
    
    // Launch kernel
    myKernel<<<256, 256>>>(d_data);
    
    // Check for kernel launch errors
    gpuErrchk(cudaGetLastError());
    
    // Check for kernel execution errors
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaFree(d_data));
}
```

### Using cuda-memcheck
`cuda-memcheck` is a powerful tool for detecting memory errors:

```bash
# Basic memory checking
cuda-memcheck ./my_cuda_app

# Check for memory leaks
cuda-memcheck --tool memcheck --leak-check full ./my_cuda_app

# Check for race conditions
cuda-memcheck --tool racecheck ./my_cuda_app

# Check for mismatched memory accesses
cuda-memcheck --tool memcheck --track-unused-memory yes ./my_cuda_app
```

### Example of error-prone code and debugging:
```cuda
// Problematic kernel with potential out-of-bounds access
__global__ void buggyKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BUG: No bounds checking - potential out-of-bounds access
    data[idx] = data[idx] * 2.0f;  // Could access invalid memory
}

// Corrected version with bounds checking
__global__ void correctedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIXED: Add bounds checking
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

## CUDA-GDB Debugger

### Using CUDA-GDB
CUDA-GDB extends GDB with GPU debugging capabilities:

```bash
# Start debugging
cuda-gdb ./my_cuda_app

# Common CUDA-GDB commands:
(cuda-gdb) break myKernel
(cuda-gdb) run
(cuda-gdb) cuda thread (1,0,0)  # Switch to specific thread
(cuda-gdb) print threadIdx.x     # Print thread variable
(cuda-gdb) cuda block (0,0,0)   # Switch to specific block
(cuda-gdb) info cuda kernels     # List kernels
```

### Debugging with breakpoints:
```cuda
// Kernel with debugging aids
__global__ void debugKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        // Place breakpoint here for debugging
        __debugbreak();  // Only in debug builds
    }
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

## Nsight Compute Profiler

### Basic Profiling
Nsight Compute provides detailed kernel performance analysis:

```bash
# Basic profiling
ncu ./my_cuda_app

# Profile specific kernels
ncu --kernel-name "myKernel" ./my_cuda_app

# Collect specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.bytes_per_second ./my_cuda_app

# Export results
ncu --export profile_results ./my_cuda_app
```

### Programmatic profiling with NVTX:
```cuda
#include <nvToolsExt.h>

void profiledSection() {
    nvtxRangePush("Data Processing");
    
    // Your CUDA code here
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    nvtxRangePush("Kernel Execution");
    myKernel<<<256, 256>>>(d_data);
    nvtxRangePop(); // End "Kernel Execution"
    
    cudaDeviceSynchronize();
    nvtxRangePop(); // End "Data Processing"
}

// Mark specific events
void markEvent() {
    nvtxMark("Starting computation");
    // ... computation ...
    nvtxMark("Computation completed");
}
```

## CUDA Launch Bounds for Debugging

### Using Launch Bounds
Launch bounds can help identify register pressure issues:

```cuda
// Specify maximum threads per block and minimum blocks per SM
__global__ void __launch_bounds__(256, 4)
boundedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
```

## Static Analysis Tools

### Using nvcc Compiler Flags
Enable comprehensive error checking:

```bash
# Enable all warnings
nvcc -Xcompiler -Wall -Xcompiler -Wextra -o app app.cu

# Enable pedantic checks
nvcc -Xcompiler -pedantic -o app app.cu

# Enable bounds checking (for debugging)
nvcc -g -G -o debug_app app.cu  # Debug symbols and device debug info

# Enable line info for profilers
nvcc -lineinfo -o profile_app app.cu
```

## Debugging Synchronization Issues

### Identifying Race Conditions
```cuda
// Problematic code with race condition
__global__ void raceConditionKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n-1) {
        // RACE CONDITION: Multiple threads may modify adjacent elements
        data[idx] = data[idx] + data[idx+1];
    }
}

// Fixed version with proper synchronization
__global__ void synchronizedKernel(float *data, float *temp, int n) {
    __shared__ float s_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // Load data to shared memory
    if (idx < n) {
        s_data[local_idx] = data[idx];
    }
    if (idx + 1 < n) {
        s_data[local_idx + 1] = data[idx + 1];
    }
    __syncthreads();
    
    // Perform computation safely
    if (idx < n-1) {
        data[idx] = s_data[local_idx] + s_data[local_idx + 1];
    }
}
```

## Memory Debugging Techniques

### Detecting Memory Leaks
```cuda
class CudaMemoryManager {
private:
    std::vector<void*> allocated_ptrs;
    
public:
    void* malloc(size_t size) {
        void* ptr;
        cudaMalloc(&ptr, size);
        allocated_ptrs.push_back(ptr);
        return ptr;
    }
    
    void free(void* ptr) {
        cudaFree(ptr);
        allocated_ptrs.erase(
            std::remove(allocated_ptrs.begin(), allocated_ptrs.end(), ptr),
            allocated_ptrs.end()
        );
    }
    
    ~CudaMemoryManager() {
        if (!allocated_ptrs.empty()) {
            printf("WARNING: Memory leak detected! %zu pointers not freed\n", 
                   allocated_ptrs.size());
        }
    }
};
```

### Unified Memory Debugging
```cuda
void unifiedMemoryDebugging() {
    float *data;
    cudaMallocManaged(&data, 1024 * sizeof(float));
    
    // Initialize on CPU
    for (int i = 0; i < 1024; i++) {
        data[i] = i * 1.0f;
    }
    
    // Prefetch to GPU
    cudaMemPrefetchAsync(data, 1024 * sizeof(float), 0); // GPU device 0
    
    // Launch kernel
    processOnGPU<<<256, 256>>>(data, 1024);
    cudaDeviceSynchronize();
    
    // Prefetch back to CPU
    cudaMemPrefetchAsync(data, 1024 * sizeof(float), cudaCpuDeviceId);
    
    // Use on CPU
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %f\n", i, data[i]);
    }
    
    cudaFree(data);
}
```

## Profiling with Nsight Systems

### Timeline Analysis
Nsight Systems provides system-wide performance analysis:

```bash
# Basic timeline profiling
nsys profile ./my_cuda_app

# Profile for specific duration
nsys profile --duration=10 ./my_cuda_app

# Capture specific CUDA APIs
nsys profile --trace=cuda,nvtx ./my_cuda_app

# Generate report
nsys profile --export=sqlite ./my_cuda_app
```

## Performance Metrics to Monitor

### Key Metrics for Optimization
```cuda
// Example kernel with performance considerations
__global__ void performanceAwareKernel(float *input, float *output, int n) {
    // Calculate occupancy
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Memory access pattern: coalesced access
        float val = input[idx];
        
        // Arithmetic intensity: balance computation and memory
        val = val * val + sqrtf(val) + expf(-val);
        
        // Write result
        output[idx] = val;
    }
}
```

### Common Performance Issues:
1. **Low occupancy** - Not enough active warps
2. **Memory bandwidth limitations** - Slow memory access
3. **Divergent branching** - Threads in warp taking different paths
4. **Register pressure** - Too many registers per thread
5. **Shared memory bank conflicts**

## Debugging Best Practices

### Defensive Programming
```cuda
// Utility function for checking CUDA errors
const char* getCudaErrMsg() {
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}

// Wrapper for safer kernel launches
template<typename KernelFunc, typename... Args>
void safeLaunchKernel(KernelFunc kernel, dim3 grid, dim3 block, 
                     size_t sharedMem = 0, cudaStream_t stream = 0,
                     Args... args) {
    kernel<<<grid, block, sharedMem, stream>>>(args...);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Example usage
void exampleUsage() {
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    dim3 block(256);
    dim3 grid((1024 + block.x - 1) / block.x);
    
    safeLaunchKernel(myKernel, grid, block, 0, 0, d_data);
    
    cudaDeviceSynchronize();
    cudaFree(d_data);
}
```

## Profiling Example

### Complete profiling workflow:
```cuda
#include <nvtx3/nvToolsExt.h>

void profiledApplication() {
    const int N = 1000000;
    const size_t size = N * sizeof(float);
    
    nvtxRangePush("Initialization");
    float *h_data = (float*)malloc(size);
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.1f;
    }
    nvtxRangePop(); // Initialization
    
    nvtxRangePush("Data Transfer");
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    nvtxRangePop(); // Data Transfer
    
    nvtxRangePush("Kernel Execution");
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    processKernel<<<grid, block>>>(d_data, N);
    cudaDeviceSynchronize();
    nvtxRangePop(); // Kernel Execution
    
    nvtxRangePush("Result Transfer");
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    nvtxRangePop(); // Result Transfer
    
    // Cleanup
    free(h_data);
    cudaFree(d_data);
}

__global__ void processKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * data[idx] + 1.0f;
    }
}
```

## Summary
This chapter covered essential debugging and profiling techniques for CUDA applications:
- Runtime error checking and validation
- Using cuda-memcheck for memory error detection
- CUDA-GDB for interactive debugging
- Nsight Compute for detailed kernel profiling
- Nsight Systems for timeline analysis
- NVTX for programmatic profiling
- Common debugging patterns and best practices
- Performance metrics to monitor
- Defensive programming techniques

Effective debugging and profiling are crucial for developing robust and high-performance CUDA applications. The next chapter will explore multi-GPU programming techniques.