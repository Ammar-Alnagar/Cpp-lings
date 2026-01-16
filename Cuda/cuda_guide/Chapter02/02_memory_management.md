# Chapter 2: Memory Management in CUDA

## Overview
Memory management is a critical aspect of CUDA programming. Understanding the different types of memory and how to efficiently use them is essential for achieving optimal performance. This chapter covers the various memory spaces available in CUDA and techniques for managing them effectively.

## Types of Memory in CUDA

### 1. Global Memory
Global memory is the largest memory space available on the GPU. It's accessible by all threads and persists for the entire lifetime of the application.

#### Characteristics:
- Largest capacity (several GB)
- Highest latency
- Cached in L1 and L2 cache (on newer architectures)
- Accessible by all threads and the host

#### Example:
```cuda
__global__ void globalMemExample(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Accessing global memory
        data[idx] = data[idx] * 2.0f;
    }
}
```

### 2. Shared Memory
Shared memory is a small, fast memory space shared among threads within a block. It's much faster than global memory and can be used for data reuse and inter-thread communication.

#### Characteristics:
- Small capacity (typically 16-48 KB per block)
- Very low latency
- Shared among threads in the same block
- Explicitly managed by the programmer

#### Example:
```cuda
__global__ void sharedMemExample(float *input, float *output, int n) {
    // Declare shared memory
    __shared__ float sData[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load data from global memory to shared memory
    if (tid < n) {
        sData[local_tid] = input[tid];
    } else {
        sData[local_tid] = 0.0f;
    }
    
    __syncthreads(); // Synchronize threads in the block
    
    // Perform computation using shared memory
    if (tid < n) {
        output[tid] = sData[local_tid] * 2.0f;
    }
}
```

### 3. Constant Memory
Constant memory is read-only memory cached for uniform access across warps. It's optimized for data that remains constant during kernel execution.

#### Characteristics:
- Small capacity (64 KB)
- Read-only from device
- Cached for uniform access
- Optimized for broadcast scenarios

#### Example:
```cuda
// Declare constant memory
__constant__ float coefficients[32];

__global__ void constantMemExample(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Access constant memory
        output[idx] = input[idx] * coefficients[idx % 32];
    }
}

// Host code to copy data to constant memory
void setupConstants() {
    float h_coeffs[32];
    // Initialize coefficients
    for (int i = 0; i < 32; i++) {
        h_coeffs[i] = i * 0.5f;
    }
    
    // Copy to constant memory
    cudaMemcpyToSymbol(coefficients, h_coeffs, sizeof(h_coeffs));
}
```

### 4. Texture Memory
Texture memory is read-only memory with special caching behavior. It provides interpolation and boundary handling capabilities.

#### Example:
```cuda
// Texture reference (legacy API)
texture<float, 1, cudaReadModeElementType> tex;

__global__ void textureMemExample(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Access texture memory with interpolation
        output[idx] = tex1D(tex, idx + 0.5f);
    }
}
```

### 5. Local Memory
Local memory is per-thread memory used for local variables that don't fit in registers.

#### Characteristics:
- Per-thread storage
- Actually located in global memory
- Used when register space is insufficient
- High latency

## Memory Allocation and Transfer

### Allocating Memory
```cuda
// Allocate host memory
float *h_data = (float*)malloc(N * sizeof(float));

// Allocate device memory
float *d_data;
cudaMalloc((void**)&d_data, N * sizeof(float));

// Allocate pinned host memory (faster transfers)
float *h_pinned_data;
cudaMallocHost((void**)&h_pinned_data, N * sizeof(float));
```

### Memory Transfer Operations
```cuda
// Host to Device
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// Device to Device
cudaMemcpy(d_dest, d_src, N * sizeof(float), cudaMemcpyDeviceToDevice);

// Asynchronous transfers (with streams)
cudaMemcpyAsync(d_data, h_data, N * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
```

## Unified Memory
Unified Memory provides a single memory address space accessible by both CPU and GPU, simplifying memory management.

```cuda
#include <cuda_runtime.h>

__global__ void unifiedMemExample(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    const int N = 1024;
    float *data;
    
    // Allocate unified memory
    cudaMallocManaged(&data, N * sizeof(float));
    
    // Initialize data on CPU
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    unifiedMemExample<<<gridSize, blockSize>>>(data, N);
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Data can be accessed directly from CPU
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %f\n", i, data[i]);
    }
    
    // Free unified memory
    cudaFree(data);
    
    return 0;
}
```

## Memory Optimization Techniques

### 1. Coalesced Memory Access
For optimal performance, threads in a warp should access consecutive memory addresses:

```cuda
// GOOD: Coalesced access
__global__ void coalescedAccess(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Consecutive threads access consecutive memory locations
        output[idx] = input[idx] * 2.0f;
    }
}

// BAD: Strided access (poor performance)
__global__ void stridedAccess(float *input, float *output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Threads access memory with stride, causing multiple transactions
        output[idx] = input[idx * stride] * 2.0f;
    }
}
```

### 2. Using Shared Memory for Reduction
```cuda
__global__ void reductionWithSharedMem(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result back to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### 3. Memory Padding for Alignment
```cuda
// Structure that might cause misalignment
struct BadStruct {
    char a;      // 1 byte
    int b;       // 4 bytes
    double c;    // 8 bytes
};

// Better: Padded structure for alignment
struct GoodStruct {
    char a;      // 1 byte
    char pad[3]; // 3 bytes padding
    int b;       // 4 bytes
    double c;    // 8 bytes
};
```

## Memory Bandwidth and Latency Considerations

### Bank Conflicts in Shared Memory
Shared memory is divided into banks. Accessing different addresses in the same bank by different threads causes serialization:

```cuda
// POTENTIAL BANK CONFLICT
__global__ void bankConflictExample(float *data) {
    __shared__ float s[32][33]; // Pad to avoid conflicts
    
    int tid = threadIdx.x;
    // This access pattern could cause bank conflicts
    s[threadIdx.x][threadIdx.x] = data[threadIdx.x];
}

// AVOIDING BANK CONFLICTS
__global__ void noBankConflictExample(float *data) {
    __shared__ float s[32][33]; // 33 elements per row avoids conflicts
    
    int tid = threadIdx.x;
    // This access pattern avoids bank conflicts due to padding
    s[threadIdx.x][threadIdx.x] = data[threadIdx.x];
}
```

## Memory Error Checking
Always check for memory errors in CUDA applications:

```cuda
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main() {
    float *d_data;
    gpuErrchk(cudaMalloc(&d_data, 1024 * sizeof(float)));
    
    // ... rest of the code ...
    
    gpuErrchk(cudaFree(d_data));
    return 0;
}
```

## Summary
This chapter covered the essential aspects of memory management in CUDA:
- Different types of memory and their characteristics
- Memory allocation and transfer techniques
- Unified memory for simplified programming
- Optimization strategies for better performance
- Common pitfalls and how to avoid them

Effective memory management is crucial for achieving high performance in CUDA applications. The next chapter will explore thread programming in detail.