#pragma once
#include <cuda_runtime.h>

namespace hpc::gpu {

template<typename T, unsigned int BLOCK_SIZE>
__global__ void reduction_kernel(const T* input, T* output, size_t n) {
    extern __shared__ T sdata[];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    size_t grid_size = blockDim.x * 2 * gridDim.x;

    T sum = T{0};

    while (i < n) {
        sum += input[i];
        if (i + blockDim.x < n) {
            sum += input[i + blockDim.x];
        }
        i += grid_size;
    }

    sdata[tid] = sum;
    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) {
        volatile T* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template<typename T, int TILE_SIZE>
__global__ void matrix_multiply_tiled_kernel(const T* A, const T* B, T* C,
                                               size_t M, size_t N, size_t K) {
    __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = T{0};

    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = 
                A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = T{0};
        }

        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = T{0};
        }

        __syncthreads();

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

} // namespace hpc::gpu
