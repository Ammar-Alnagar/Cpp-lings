# Hands-On Project 1: Vector Operations with CUDA

## Overview
This project will teach you the fundamentals of CUDA programming by implementing various vector operations. You'll learn about memory management, kernel launches, and basic parallel operations.

## Learning Objectives
- Understand CUDA memory management (malloc, memcpy, free)
- Learn how to launch kernels with proper grid/block dimensions
- Implement basic vector operations in parallel
- Practice error checking in CUDA applications

## Project Structure
```
project1_vector_operations/
├── README.md
├── vector_ops.cu
├── Makefile
└── solution/
    └── vector_ops_solution.cu
```

## Step-by-Step Guide

### Step 1: Set up the basic structure
Create a file called `vector_ops.cu` with the basic structure:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// TODO: Implement the following functions:
// 1. vectorAdd - Add two vectors element-wise
// 2. vectorScale - Multiply each element by a scalar
// 3. vectorDotProduct - Compute dot product of two vectors
// 4. vectorMagnitude - Compute magnitude of a vector

// Helper function for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// TODO: Implement vector addition kernel
__global__ void vectorAdd(/* parameters */) {
    // Calculate the global thread index
    // Check bounds
    // Perform addition: c[i] = a[i] + b[i]
}

// TODO: Implement vector scaling kernel
__global__ void vectorScale(/* parameters */) {
    // Calculate the global thread index
    // Check bounds
    // Perform scaling: a[i] = a[i] * scalar
}

// TODO: Implement dot product kernel (reduction)
__global__ void vectorDotProduct(/* parameters */) {
    // Use shared memory for reduction
    // Each thread computes a[i] * b[i]
    // Reduce within block
    // Store partial result
}

// TODO: Implement magnitude calculation kernel
__global__ void vectorMagnitude(/* parameters */) {
    // Calculate squared values
    // Use reduction to sum all squares
    // Take square root
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // TODO: Allocate host memory
    float *h_a = /* allocate */;
    float *h_b = /* allocate */;
    float *h_c = /* allocate */;
    float *h_result = /* allocate */;
    
    // TODO: Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // TODO: Allocate device memory
    float *d_a, *d_b, *d_c;
    // cudaMalloc for each array
    
    // TODO: Copy data from host to device
    // cudaMemcpy from host to device
    
    // TODO: Set up execution configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // TODO: Launch vector addition kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // TODO: Copy result back to host
    // cudaMemcpy from device to host
    
    // TODO: Verify results
    printf("First 10 results of vector addition: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_c[i]);
    }
    printf("\n");
    
    // TODO: Implement vector scaling
    float scale_factor = 2.5f;
    // Launch vectorScale kernel
    // Copy result back and verify
    
    // TODO: Implement dot product calculation
    // You'll need to implement a reduction approach
    float dot_product_result;
    // Launch dot product kernel
    // Copy result back
    
    // TODO: Implement magnitude calculation
    float magnitude_result;
    // Launch magnitude kernel
    // Copy result back
    
    // TODO: Cleanup allocated memory
    // cudaFree for device memory
    // free for host memory
    
    printf("Project 1 completed successfully!\n");
    
    return 0;
}
```

### Step 2: Implement Vector Addition
Complete the `vectorAdd` kernel:

```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Step 3: Implement Vector Scaling
Complete the `vectorScale` kernel:

```cuda
__global__ void vectorScale(float *a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        a[idx] *= scalar;
    }
}
```

### Step 4: Implement Dot Product with Reduction
Complete the `vectorDotProduct` kernel:

```cuda
__global__ void vectorDotProduct(float *a, float *b, float *result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}
```

### Step 5: Complete the Main Function
Fill in the remaining parts of the main function with proper CUDA calls.

### Step 6: Create a Makefile
Create a `Makefile`:

```makefile
CC = nvcc
CFLAGS = -O3 -arch=sm_50
TARGET = vector_ops
SOURCE = vector_ops.cu

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

.PHONY: clean
```

### Step 7: Test Your Implementation
1. Compile your code: `make`
2. Run the executable: `./vector_ops`
3. Verify that the results are correct
4. Compare with the solution if needed

## Challenge Extensions
1. Implement vector subtraction
2. Add support for different data types (double, int)
3. Optimize the dot product kernel further
4. Implement a fused multiply-add operation

## Solution
A complete solution is provided in the `solution/` directory for reference after you've attempted the implementation yourself.