#pragma once
#include <cuda_runtime.h>

namespace hpc::gpu {

template<typename T>
__global__ void vector_add_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void vector_subtract_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

template<typename T>
__global__ void vector_scale_kernel(const T* a, T scalar, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

template<typename T>
__global__ void matrix_init_kernel(T* data, size_t rows, size_t cols, T value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;
    if (idx < total) {
        data[idx] = value;
    }
}

template<typename T>
__global__ void matrix_transpose_kernel(const T* input, T* output, 
                                       size_t rows, size_t cols) {
    __shared__ T tile[32][32];

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template<typename T>
__global__ void matrix_multiply_simple_kernel(const T* A, const T* B, T* C,
                                               size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = T{0};
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

} // namespace hpc::gpu
