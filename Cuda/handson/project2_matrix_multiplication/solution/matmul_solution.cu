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

// Naive matrix multiplication kernel
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

// Tiled matrix multiplication kernel
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

// Optimized matrix multiplication kernel
__global__ void matmul_optimized(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory with improved coalescing
        As[ty][tx] = (row < M && (t * TILE_SIZE + tx) < K) ? 
                      A[row * K + t * TILE_SIZE + tx] : 0.0f;
        Bs[ty][tx] = ((t * TILE_SIZE + ty) < K && col < N) ? 
                      B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
        
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

int main() {
    const int M = 512;  // Rows of A and C
    const int N = 512;  // Columns of B and C
    const int K = 512;  // Columns of A and rows of B
    
    const int size_A = M * K * sizeof(float);
    const int size_B = K * N * sizeof(float);
    const int size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    // Initialize host matrices
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(i % 7);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(i % 5);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    gpuErrchk(cudaMalloc(&d_A, size_A));
    gpuErrchk(cudaMalloc(&d_B, size_B));
    gpuErrchk(cudaMalloc(&d_C, size_C));
    
    // Copy data from host to device
    gpuErrchk(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Set up execution configuration for naive implementation
    int blockSize = 16;  // Use 16x16 blocks
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
    
    // Time naive implementation
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Launch naive kernel
    matmul_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gettimeofday(&end, NULL);
    float naive_time = get_time_diff(start, end);
    
    // Set up execution configuration for tiled implementation
    dim3 tileBlock(TILE_SIZE, TILE_SIZE);
    dim3 tileGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Time tiled implementation
    gettimeofday(&start, NULL);
    
    // Launch tiled kernel
    matmul_tiled<<<tileGrid, tileBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gettimeofday(&end, NULL);
    float tiled_time = get_time_diff(start, end);
    
    // Time optimized implementation
    gettimeofday(&start, NULL);
    
    // Launch optimized kernel
    matmul_optimized<<<tileGrid, tileBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gettimeofday(&end, NULL);
    float opt_time = get_time_diff(start, end);
    
    // Copy result back for verification
    gpuErrchk(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    printf("Matrix size: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Naive implementation time: %.4f seconds\n", naive_time);
    printf("Tiled implementation time: %.4f seconds\n", tiled_time);
    printf("Optimized implementation time: %.4f seconds\n", opt_time);
    printf("Speedup with tiling: %.2fx\n", naive_time / tiled_time);
    printf("Speedup with optimization: %.2fx\n", naive_time / opt_time);
    
    // Print first few results for verification
    printf("First 5x5 elements of result matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    
    // Cleanup allocated memory
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}