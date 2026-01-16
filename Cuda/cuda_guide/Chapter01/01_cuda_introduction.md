# Chapter 1: Introduction to CUDA

## Overview
CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It enables developers to use NVIDIA GPUs for general-purpose computing, allowing massive parallelization of compute-intensive applications.

## GPU Architecture Basics
Modern GPUs contain hundreds or thousands of cores designed for parallel processing. Unlike CPUs which are optimized for sequential tasks, GPUs excel at executing many similar operations simultaneously. This makes them ideal for data-parallel computations.

### Key Concepts:
- **Streaming Multiprocessors (SMs)**: The basic computational units of a GPU
- **CUDA Cores**: Individual processing elements within SMs
- **Warps**: Groups of 32 threads that execute in lockstep
- **Grid**: Collection of thread blocks
- **Blocks**: Collection of threads that can cooperate via shared memory

## CUDA Programming Model
CUDA extends C/C++ with keywords and syntax to define functions that run on the GPU:

- `__global__`: Functions called from CPU, executed on GPU (kernels)
- `__device__`: Functions called from GPU, executed on GPU
- `__host__`: Functions called from CPU, executed on CPU (default)

## Hello World Example
Let's start with a simple CUDA program that prints "Hello from GPU!" from multiple threads:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

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
```

### Explanation:
1. `__global__ void helloFromGPU(int n)`: This is a kernel function that runs on the GPU
2. `blockIdx.x * blockDim.x + threadIdx.x`: Calculates the unique thread index
3. `helloFromGPU<<<gridSize, blockSize>>>(numThreads)`: Launches the kernel with specified grid and block dimensions
4. `cudaDeviceSynchronize()`: Ensures GPU execution completes before CPU continues

## Vector Addition Example
A more practical example demonstrates how to add two vectors using GPU:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Kernel function to add two vectors
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Host variables
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;  // Device variables
    
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### Key Steps in CUDA Programming:
1. **Allocate device memory** using `cudaMalloc`
2. **Transfer data** from host to device using `cudaMemcpy`
3. **Launch kernels** with appropriate grid and block dimensions
4. **Transfer results** back from device to host using `cudaMemcpy`
5. **Free device memory** using `cudaFree`

## Compilation
To compile CUDA programs, use the NVIDIA compiler `nvcc`:

```bash
nvcc -o hello hello.cu
./hello
```

## Grid and Block Configuration
Understanding how to configure grids and blocks is crucial for efficient CUDA programming:

- `threadIdx.x`: Thread index within a block (0 to blockDim.x-1)
- `blockIdx.x`: Block index within a grid (0 to gridDim.x-1)
- `blockDim.x`: Number of threads per block
- Global thread ID: `blockIdx.x * blockDim.x + threadIdx.x`

## Memory Hierarchy
CUDA provides different types of memory:
- **Global Memory**: Large, high-latency memory accessible by all threads
- **Shared Memory**: Small, fast memory shared among threads in a block
- **Local Memory**: Per-thread memory for local variables
- **Constant Memory**: Read-only memory cached for uniform access
- **Texture Memory**: Read-only memory with spatial locality caching

## Summary
This chapter introduced the basics of CUDA programming including:
- The CUDA programming model
- Basic kernel structure
- Memory allocation and transfer
- Grid and block configuration
- A simple vector addition example

The next chapter will dive deeper into memory management in CUDA.