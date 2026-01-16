#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// Map pattern kernel
__global__ void mapKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = input[idx] * input[idx] + 2.0f * input[idx] + 1.0f;
    }
}

// Basic reduction kernel
__global__ void reduceBasic(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
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
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized reduction kernel
__global__ void reduceOptimized(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Perform first level of reduction
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    sdata[tid] += (i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Unroll the last warp
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Histogram kernel
__global__ void histogramKernel(unsigned char *input, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unsigned char value = input[idx];
        atomicAdd(&histogram[value], 1);
    }
}

// Matrix multiplication kernel
__global__ void matMulBasic(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication kernel
__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && (t * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < N && col < N) {
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
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// 2D 5-point stencil kernel
__global__ void stencil5pt(float *input, float *output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        float center = input[row * width + col];
        float north = input[(row - 1) * width + col];
        float south = input[(row + 1) * width + col];
        float east = input[row * width + col + 1];
        float west = input[row * width + col - 1];
        
        // 5-point stencil computation
        output[row * width + col] = 0.2f * (center + north + south + east + west);
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
    const int M = 64; // For matrix operations
    const int size = N * sizeof(float);
    const int mat_size = M * M * sizeof(float);
    
    printf("=== Parallel Programming Patterns Example ===\n");
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    float *h_matrixA = (float*)malloc(mat_size);
    float *h_matrixB = (float*)malloc(mat_size);
    float *h_matrixC = (float*)malloc(mat_size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 0.1f;
    }
    
    // Initialize matrices
    for (int i = 0; i < M * M; i++) {
        h_matrixA[i] = i * 0.1f;
        h_matrixB[i] = (M * M - i) * 0.1f;
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_matrixA, *d_matrixB, *d_matrixC;
    gpuErrchk(cudaMalloc(&d_input, size));
    gpuErrchk(cudaMalloc(&d_output, size));
    gpuErrchk(cudaMalloc(&d_matrixA, mat_size));
    gpuErrchk(cudaMalloc(&d_matrixB, mat_size));
    gpuErrchk(cudaMalloc(&d_matrixC, mat_size));
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_matrixA, h_matrixA, mat_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_matrixB, h_matrixB, mat_size, cudaMemcpyHostToDevice));
    
    printf("Memory allocated and initialized.\n");
    
    // Map pattern example
    printf("\n=== Map Pattern Example ===\n");
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    mapKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    printf("First 10 values after mapping:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");
    
    // Reduction pattern example
    printf("\n=== Reduction Pattern Example ===\n");
    
    float *d_reduced, *d_final_result;
    gpuErrchk(cudaMalloc(&d_reduced, gridSize * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_final_result, sizeof(float)));
    
    reduceOptimized<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_reduced, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Perform final reduction on CPU
    float *h_reduced = (float*)malloc(gridSize * sizeof(float));
    gpuErrchk(cudaMemcpy(h_reduced, d_reduced, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    float final_sum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        final_sum += h_reduced[i];
    }
    
    printf("Sum of all elements: %.2f\n", final_sum);
    
    // Matrix multiplication example
    printf("\n=== Matrix Multiplication Example ===\n");
    
    dim3 mat_block(TILE_SIZE, TILE_SIZE);
    dim3 mat_grid((M + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matMulTiled<<<mat_grid, mat_block>>>(d_matrixA, d_matrixB, d_matrixC, M);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_matrixC, d_matrixC, mat_size, cudaMemcpyDeviceToHost));
    
    printf("First 10 values of result matrix:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_matrixC[i]);
    }
    printf("\n");
    
    // Histogram example
    printf("\n=== Histogram Example ===\n");
    
    unsigned char *h_char_input = (unsigned char*)malloc(N * sizeof(unsigned char));
    unsigned char *d_char_input;
    int *d_histogram, *h_histogram;
    
    // Initialize character data
    for (int i = 0; i < N; i++) {
        h_char_input[i] = i % 256;
    }
    
    gpuErrchk(cudaMalloc(&d_char_input, N * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&d_histogram, 256 * sizeof(int)));
    
    h_histogram = (int*)malloc(256 * sizeof(int));
    
    // Initialize histogram to zero
    for (int i = 0; i < 256; i++) {
        h_histogram[i] = 0;
    }
    
    gpuErrchk(cudaMemcpy(d_char_input, h_char_input, N * sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_histogram, h_histogram, 256 * sizeof(int), cudaMemcpyHostToDevice));
    
    histogramKernel<<<gridSize, blockSize>>>(d_char_input, d_histogram, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Histogram of first 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("Value %d: %d times\n", i, h_histogram[i]);
    }
    
    // Stencil pattern example
    printf("\n=== Stencil Pattern Example ===\n");
    
    const int STENCIL_N = 128;
    const int stencil_size = STENCIL_N * STENCIL_N * sizeof(float);
    
    float *h_stencil_input = (float*)malloc(stencil_size);
    float *h_stencil_output = (float*)malloc(stencil_size);
    float *d_stencil_input, *d_stencil_output;
    
    // Initialize stencil input
    for (int i = 0; i < STENCIL_N * STENCIL_N; i++) {
        h_stencil_input[i] = (float)(i % 100);
    }
    
    gpuErrchk(cudaMalloc(&d_stencil_input, stencil_size));
    gpuErrchk(cudaMalloc(&d_stencil_output, stencil_size));
    
    gpuErrchk(cudaMemcpy(d_stencil_input, h_stencil_input, stencil_size, cudaMemcpyHostToDevice));
    
    dim3 stencil_block(16, 16);
    dim3 stencil_grid((STENCIL_N + 15) / 16, (STENCIL_N + 15) / 16);
    
    stencil5pt<<<stencil_grid, stencil_block>>>(d_stencil_input, d_stencil_output, STENCIL_N, STENCIL_N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_stencil_output, d_stencil_output, stencil_size, cudaMemcpyDeviceToHost));
    
    printf("First 10 values after stencil operation:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_stencil_output[i]);
    }
    printf("\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixC);
    free(h_reduced);
    free(h_char_input);
    free(h_histogram);
    free(h_stencil_input);
    free(h_stencil_output);
    
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));
    gpuErrchk(cudaFree(d_matrixA));
    gpuErrchk(cudaFree(d_matrixB));
    gpuErrchk(cudaFree(d_matrixC));
    gpuErrchk(cudaFree(d_reduced));
    gpuErrchk(cudaFree(d_final_result));
    gpuErrchk(cudaFree(d_char_input));
    gpuErrchk(cudaFree(d_histogram));
    gpuErrchk(cudaFree(d_stencil_input));
    gpuErrchk(cudaFree(d_stencil_output));
    
    printf("\nAll parallel programming pattern examples completed successfully.\n");
    
    return 0;
}