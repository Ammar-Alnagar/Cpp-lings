# Chapter 8: CUDA Libraries and Integration with Other Languages

## Overview
CUDA provides a rich ecosystem of libraries that accelerate common computational tasks and facilitate integration with other programming languages. This chapter covers the major CUDA libraries and approaches for using CUDA from different programming environments.

## CUDA Math Libraries

### cuBLAS (CUDA Basic Linear Algebra Subroutines)
cuBLAS provides GPU-accelerated implementations of BLAS routines.

```cuda
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

void cublasExample() {
    const int N = 256;
    
    // Host data
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    
    // Device data
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform SAXPY: y = alpha*x + y
    const float alpha = 2.0f;
    cublasSaxpy(handle, N, &alpha, d_A, 1, d_B, 1);
    
    // Copy result back
    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("First 10 results of SAXPY: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_B[i]);
    }
    printf("\n");
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
```

### cuFFT (CUDA Fast Fourier Transform)
cuFFT provides GPU-accelerated FFT computations.

```cuda
#include <cufft.h>
#include <cuda_runtime.h>

void cufftExample() {
    const int N = 256;
    
    // Host data
    float *h_signal = (float*)malloc(N * 2 * sizeof(float)); // Complex numbers (real, imag)
    
    // Initialize signal (real part = sine wave, imaginary part = 0)
    for (int i = 0; i < N; i++) {
        h_signal[2*i] = sinf(2.0f * M_PI * i / N);     // Real part
        h_signal[2*i + 1] = 0.0f;                       // Imaginary part
    }
    
    // Device data
    float *d_signal;
    cudaMalloc(&d_signal, N * 2 * sizeof(float));
    cudaMemcpy(d_signal, h_signal, N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);  // Complex-to-complex 1D FFT
    
    // Execute FFT
    cufftExecC2C(plan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD);
    
    // Copy result back
    cudaMemcpy(h_signal, d_signal, N * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("First 5 FFT results (real, imag): ");
    for (int i = 0; i < 5; i++) {
        printf("(%.2f, %.2f) ", h_signal[2*i], h_signal[2*i + 1]);
    }
    printf("\n");
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_signal);
    free(h_signal);
}
```

### cuSPARSE (CUDA Sparse Matrix Library)
cuSPARSE handles sparse matrix operations.

```cuda
#include <cusparse.h>
#include <cuda_runtime.h>

void cusparseExample() {
    // Example: sparse matrix-vector multiplication (SpMV)
    const int m = 4;  // Number of rows
    const int nnz = 9; // Number of non-zero elements
    
    // Host data for sparse matrix in CSR format
    int *h_csrRowPtr = (int*){0, 3, 4, 7, 9};  // Row pointers
    int *h_csrColInd = (int[]){0, 2, 3, 1, 0, 2, 3, 1, 3};  // Column indices
    float *h_csrVal = (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};  // Values
    
    // Dense vector
    float *h_x = (float[]){1.0f, 2.0f, 3.0f, 4.0f};
    float *h_y = (float*)malloc(m * sizeof(float));
    
    // Device data
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_y;
    
    cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, nnz * sizeof(int));
    cudaMalloc(&d_csrVal, nnz * sizeof(float));
    cudaMalloc(&d_x, m * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, m * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create matrix descriptor
    cusparseMatDescr_t descr = 0;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    // Perform SpMV: y = alpha*A*x + beta*y
    const float alpha = 1.0f, beta = 0.0f;
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   m, m, nnz, &alpha, descr,
                   d_csrVal, d_csrRowPtr, d_csrColInd,
                   d_x, &beta, d_y);
    
    // Copy result back
    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("SpMV result: ");
    for (int i = 0; i < m; i++) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");
    
    // Cleanup
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_y);
}
```

### cuRAND (CUDA Random Number Generation)
cuRAND generates random numbers on the GPU.

```cuda
#include <curand.h>
#include <cuda_runtime.h>

__global__ void setup_kernel(curandState *state, unsigned long seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void generate_kernel(curandState *state, float *result, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        result[id] = curand_uniform(&state[id]);  // Generate uniform [0,1) random numbers
    }
}

void curandExample() {
    const int N = 1024;
    
    // Device data
    curandState *d_state;
    float *d_result, *h_result;
    
    cudaMalloc(&d_state, N * sizeof(curandState));
    cudaMalloc(&d_result, N * sizeof(float));
    h_result = (float*)malloc(N * sizeof(float));
    
    // Setup random states
    setup_kernel<<<(N + 255) / 256, 256>>>(d_state, time(NULL), N);
    
    // Generate random numbers
    generate_kernel<<<(N + 255) / 256, 256>>>(d_state, d_result, N);
    
    // Copy results back
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("First 10 random numbers: ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", h_result[i]);
    }
    printf("\n");
    
    // Cleanup
    cudaFree(d_state);
    cudaFree(d_result);
    free(h_result);
}
```

### Thrust (C++ Template Library)
Thrust provides high-level parallel algorithms.

```cuda
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <iostream>

struct square {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

void thrustExample() {
    const int N = 1000;
    
    // Create host vector
    thrust::host_vector<float> h_vec(N);
    for (int i = 0; i < N; i++) {
        h_vec[i] = i * 0.1f;
    }
    
    // Copy to device
    thrust::device_vector<float> d_vec = h_vec;
    
    // Transform: square each element
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), square());
    
    // Reduce: sum all elements
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    
    // Sort the vector
    thrust::sort(d_vec.begin(), d_vec.end());
    
    // Copy result back to host
    thrust::host_vector<float> result = d_vec;
    
    printf("Sum of squares: %.2f\n", sum);
    printf("First 10 sorted values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");
}
```

## Integration with Other Languages

### Python with PyCUDA
PyCUDA allows calling CUDA kernels from Python.

```python
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel code
kernel_code = """
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)

# Get the kernel function
multiply_them = mod.get_function("multiply_them")

# Create input data
a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

# Allocate device memory
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
dest_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

# Copy data to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Launch kernel
multiply_them(dest_gpu, a_gpu, b_gpu, block=(400,1,1), grid=(1,1))

# Copy result back to host
dest = np.empty_like(a)
cuda.memcpy_dtoh(dest, dest_gpu)

print("First 10 results:", dest[:10])
```

### Python with CuPy
CuPy provides NumPy-compatible interface for GPU arrays.

```python
import cupy as cp

# Create GPU arrays
x = cp.array([1, 2, 3, 4, 5])
y = cp.array([6, 7, 8, 9, 10])

# Perform operations on GPU
z = x * y + 2
print("Result:", z)

# Matrix operations
A = cp.random.random((1000, 1000))
B = cp.random.random((1000, 1000))
C = cp.dot(A, B)  # GPU-accelerated matrix multiplication
print("Matrix multiplication completed on GPU")
```

### Python with Numba
Numba provides JIT compilation for CUDA kernels.

```python
from numba import cuda
import numpy as np

@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

# Host code
n = 100000
a = np.random.random(n).astype(np.float32)
b = np.random.random(n).astype(np.float32)
c = np.zeros_like(a)

# Copy to device
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# Launch kernel
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy result back
result = d_c.copy_to_host()
print("First 10 results:", result[:10])
```

### C++ with CUDA Runtime API
Combining C++ host code with CUDA kernels.

```cpp
// vector_add.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

class CudaVectorAdder {
public:
    void addVectors(const std::vector<float>& a, 
                   const std::vector<float>& b, 
                   std::vector<float>& result) {
        int n = a.size();
        size_t size = n * sizeof(float);
        
        // Device pointers
        float *d_a, *d_b, *d_c;
        
        // Allocate device memory
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        
        // Copy data to device
        cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        
        // Copy result back
        result.resize(n);
        cudaMemcpy(result.data(), d_c, size, cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
};

int main() {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> result;
    
    CudaVectorAdder adder;
    adder.addVectors(a, b, result);
    
    std::cout << "Result: ";
    for (float val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

## Mixed Precision Computing

### Using Half Precision (FP16)
```cuda
#include <cuda_fp16.h>

__global__ void halfPrecisionKernel(__half *input, __half *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Convert to float, perform computation, convert back
        float a = __half2float(input[idx]);
        float result = a * a + 2.0f * a + 1.0f;
        output[idx] = __float2half(result);
    }
}

void mixedPrecisionExample() {
    const int N = 1024;
    
    // Host data in half precision
    __half *h_input = (__half*)malloc(N * sizeof(__half));
    __half *h_output = (__half*)malloc(N * sizeof(__half));
    
    // Initialize with half precision values
    for (int i = 0; i < N; i++) {
        h_input[i] = __float2half(i * 0.1f);
    }
    
    // Device data
    __half *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(__half));
    cudaMalloc(&d_output, N * sizeof(__half));
    
    cudaMemcpy(d_input, h_input, N * sizeof(__half), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    halfPrecisionKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    // Copy result back
    cudaMemcpy(h_output, d_output, N * sizeof(__half), cudaMemcpyDeviceToHost);
    
    printf("First 10 half-precision results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", __half2float(h_output[i]));
    }
    printf("\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
}
```

## Tensor Core Operations

### Using Tensor Cores for Matrix Multiplication
```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <wmma.h>

using namespace nvcuda;

void tensorCoreExample() {
    // Tensor Core operations use 16x16x16 matrices with half precision
    const int N = 16;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Load matrices A and B into fragments
    // (This would typically be done in a kernel with proper memory layout)
    
    // Perform matrix multiplication: acc_frag = acc_frag + a_frag * b_frag
    // wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    
    // Note: Full Tensor Core example requires specific memory layout
    // and is typically implemented in a dedicated kernel
}
```

## Performance Optimization with Libraries

### Using cuBLAS for Optimized GEMM
```cuda
void optimizedGemm() {
    const int M = 1024, N = 1024, K = 1024;
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform matrix multiplication: C = alpha*A*B + beta*C
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, d_B, N, d_A, K,
                &beta, d_C, N);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Matrix multiplication completed using cuBLAS\n");
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
```

## Interoperability with Graphics APIs

### CUDA-OpenGL Interoperability
```cuda
#ifdef __linux__
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#include <cuda_gl_interop.h>

void cudaOpenGLInterop() {
    // This is a conceptual example - actual implementation requires OpenGL context
    /*
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vertex), 0, GL_DYNAMIC_DRAW);
    
    // Register VBO with CUDA
    cudaGraphicsResource *cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    
    // Map resource for CUDA access
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    float4 *d_positions;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &size, cuda_vbo_resource);
    
    // Launch kernel to update vertex positions
    updateVertices<<<grid, block>>>(d_positions, time);
    
    // Unmap resource
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    */
}
```

## Summary
This chapter covered:
- Major CUDA math libraries (cuBLAS, cuFFT, cuSPARSE, cuRAND)
- Thrust C++ template library for parallel algorithms
- Integration with Python (PyCUDA, CuPy, Numba)
- Integration with C++
- Mixed precision computing (FP16)
- Tensor Core operations
- Performance optimization with libraries
- Interoperability with graphics APIs

CUDA libraries significantly accelerate development by providing optimized implementations of common computational patterns. The next chapter will explore OpenACC, a directive-based parallel programming model.