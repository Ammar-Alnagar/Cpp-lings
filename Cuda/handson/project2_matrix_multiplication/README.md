# Hands-On Project 2: Matrix Multiplication with CUDA

## Overview
This project focuses on implementing optimized matrix multiplication using CUDA. You'll learn about memory access patterns, shared memory usage, and optimization techniques for better performance.

## Learning Objectives
- Implement naive matrix multiplication on GPU
- Optimize using shared memory tiling
- Understand memory coalescing
- Measure and compare performance of different implementations

## Project Structure
```
project2_matrix_multiplication/
├── README.md
├── matmul.cu
├── Makefile
└── solution/
    └── matmul_solution.cu
```

## Step-by-Step Guide

### Step 1: Set up the basic structure
Create a file called `matmul.cu` with the basic structure:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Helper function for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Helper function for timing
float get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + 1e-6*(end.tv_usec - start.tv_usec);
}

// TODO: Implement naive matrix multiplication kernel
__global__ void matmul_naive(/* parameters */) {
    // Calculate row and column indices
    // Implement triple nested loop for matrix multiplication
    // C[i][j] = sum(A[i][k] * B[k][j])
}

// TODO: Implement tiled matrix multiplication kernel
#define TILE_SIZE 16
__global__ void matmul_tiled(/* parameters */) {
    // Use shared memory tiles
    // Load tiles to shared memory
    // Implement computation using tiles
    // Handle boundary conditions
}

// TODO: Implement optimized matrix multiplication kernel
__global__ void matmul_optimized(/* parameters */) {
    // Apply optimizations learned from previous implementations
    // Consider memory access patterns, register usage, etc.
}

int main() {
    const int M = 512;  // Rows of A and C
    const int N = 512;  // Columns of B and C
    const int K = 512;  // Columns of A and rows of B
    
    const int size_A = M * K * sizeof(float);
    const int size_B = K * N * sizeof(float);
    const int size_C = M * N * sizeof(float);
    
    // TODO: Allocate host memory
    float *h_A = /* allocate */;
    float *h_B = /* allocate */;
    float *h_C = /* allocate */;
    
    // TODO: Initialize host matrices
    // Fill A and B with test values
    
    // TODO: Allocate device memory
    float *d_A, *d_B, *d_C;
    // cudaMalloc for each matrix
    
    // TODO: Copy data from host to device
    // cudaMemcpy for each matrix
    
    // TODO: Set up execution configuration for naive implementation
    int blockSize = 16;  // Use 16x16 blocks
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
    
    // TODO: Time naive implementation
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Launch naive kernel
    matmul_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gettimeofday(&end, NULL);
    float naive_time = get_time_diff(start, end);
    
    // TODO: Copy result back for verification
    // cudaMemcpy from device to host
    
    // TODO: Set up execution configuration for tiled implementation
    dim3 tileBlock(TILE_SIZE, TILE_SIZE);
    dim3 tileGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // TODO: Time tiled implementation
    gettimeofday(&start, NULL);
    
    // Launch tiled kernel
    matmul_tiled<<<tileGrid, tileBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gettimeofday(&end, NULL);
    float tiled_time = get_time_diff(start, end);
    
    // TODO: Time optimized implementation
    gettimeofday(&start, NULL);
    
    // Launch optimized kernel
    matmul_optimized<<<tileGrid, tileBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gettimeofday(&end, NULL);
    float opt_time = get_time_diff(start, end);
    
    // TODO: Verify results (optional)
    // Implement a simple CPU version to verify correctness
    
    printf("Matrix size: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Naive implementation time: %.4f seconds\n", naive_time);
    printf("Tiled implementation time: %.4f seconds\n", tiled_time);
    printf("Optimized implementation time: %.4f seconds\n", opt_time);
    printf("Speedup with tiling: %.2fx\n", naive_time / tiled_time);
    printf("Speedup with optimization: %.2fx\n", naive_time / opt_time);
    
    // TODO: Cleanup allocated memory
    // cudaFree for device memory
    // free for host memory
    
    return 0;
}
```

### Step 2: Implement Naive Matrix Multiplication
Complete the `matmul_naive` kernel:

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Step 3: Implement Tiled Matrix Multiplication
Complete the `matmul_tiled` kernel:

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Step 4: Create a Makefile
Create a `Makefile`:

```makefile
CC = nvcc
CFLAGS = -O3 -arch=sm_50
TARGET = matmul
SOURCE = matmul.cu

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

.PHONY: clean
```

### Step 5: Test Your Implementation
1. Compile your code: `make`
2. Run the executable: `./matmul`
3. Compare the performance of different implementations
4. Verify that the results are correct

## Challenge Extensions
1. Implement memory coalescing optimizations
2. Try different tile sizes and measure performance
3. Implement a version using registers instead of shared memory
4. Add support for non-square matrices
5. Implement double precision version

## Solution
A complete solution is provided in the `solution/` directory for reference after you've attempted the implementation yourself.