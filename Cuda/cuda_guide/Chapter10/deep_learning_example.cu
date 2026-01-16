#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// Basic matrix multiplication for neural networks
__global__ void matrixMul(float *A, float *B, float *C, int M, int N, int K) {
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

// Activation function: ReLU
__global__ void reluActivation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Activation function: Sigmoid
__global__ void sigmoidActivation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Optimized matrix multiplication with shared memory
__global__ void tiledMatrixMul(float *A, float *B, float *C, int M, int N, int K) {
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

// Softmax activation
__global__ void softmaxActivation(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Load data to shared memory
    if (bid * blockDim.x + tid < n) {
        sdata[tid] = input[bid * blockDim.x + tid];
    } else {
        sdata[tid] = -INFINITY;
    }
    __syncthreads();
    
    // Find maximum value
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (bid * blockDim.x + tid + s) < n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = sdata[0];
    __syncthreads();
    
    // Compute exponentials and sum
    if (bid * blockDim.x + tid < n) {
        sdata[tid] = expf(input[bid * blockDim.x + tid] - max_val);
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Sum the exponentials
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float sum_exp = sdata[0];
    __syncthreads();
    
    // Compute softmax
    if (bid * blockDim.x + tid < n) {
        output[bid * blockDim.x + tid] = sdata[tid] / sum_exp;
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
    const int INPUT_SIZE = 256;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;
    
    printf("=== Deep Learning with CUDA Example ===\n");
    
    // Allocate host memory
    float *h_input = (float*)malloc(INPUT_SIZE * sizeof(float));
    float *h_hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < INPUT_SIZE; i++) {
        h_input[i] = (float)i / INPUT_SIZE;
    }
    
    // Allocate device memory
    float *d_input, *d_weights1, *d_hidden, *d_weights2, *d_output;
    gpuErrchk(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    
    // Initialize weights randomly
    float *h_weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_weights2 = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        h_weights1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        h_weights2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    }
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Memory allocated and initialized.\n");
    
    // Forward pass: Input -> Hidden layer
    printf("\n=== Forward Pass: Input -> Hidden ===\n");
    
    dim3 mat_block(TILE_SIZE, TILE_SIZE);
    dim3 mat_grid((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE, (1 + TILE_SIZE - 1) / TILE_SIZE);
    
    // Matrix multiplication: hidden = input * weights1
    tiledMatrixMul<<<mat_grid, mat_block>>>(d_input, d_weights1, d_hidden, 1, HIDDEN_SIZE, INPUT_SIZE);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Apply activation function (ReLU)
    int blockSize = 256;
    int gridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    reluActivation<<<gridSize, blockSize>>>(d_hidden, d_hidden, HIDDEN_SIZE);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    printf("Hidden layer computed.\n");
    
    // Forward pass: Hidden -> Output layer
    printf("\n=== Forward Pass: Hidden -> Output ===\n");
    
    // Matrix multiplication: output = hidden * weights2
    dim3 out_mat_grid((OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE, (1 + TILE_SIZE - 1) / TILE_SIZE);
    tiledMatrixMul<<<out_mat_grid, mat_block>>>(d_hidden, d_weights2, d_output, 1, OUTPUT_SIZE, HIDDEN_SIZE);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Apply activation function (Softmax)
    gridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    softmaxActivation<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_output, d_output, OUTPUT_SIZE);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host
    gpuErrchk(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Output computed:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n");
    
    // Verify that softmax probabilities sum to 1
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        sum += h_output[i];
    }
    printf("Sum of output probabilities: %.4f\n", sum);
    
    // Demonstrate basic matrix multiplication
    printf("\n=== Basic Matrix Multiplication Example ===\n");
    
    const int MAT_SIZE = 64;
    float *h_A = (float*)malloc(MAT_SIZE * MAT_SIZE * sizeof(float));
    float *h_B = (float*)malloc(MAT_SIZE * MAT_SIZE * sizeof(float));
    float *h_C = (float*)malloc(MAT_SIZE * MAT_SIZE * sizeof(float));
    
    float *d_A, *d_B, *d_C;
    gpuErrchk(cudaMalloc(&d_A, MAT_SIZE * MAT_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, MAT_SIZE * MAT_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, MAT_SIZE * MAT_SIZE * sizeof(float)));
    
    // Initialize matrices
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        h_A[i] = (float)(i % MAT_SIZE) * 0.1f;
        h_B[i] = (float)((MAT_SIZE * MAT_SIZE - 1) - i) * 0.1f;
    }
    
    gpuErrchk(cudaMemcpy(d_A, h_A, MAT_SIZE * MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, h_B, MAT_SIZE * MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication
    dim3 mm_block(16, 16);
    dim3 mm_grid((MAT_SIZE + 15) / 16, (MAT_SIZE + 15) / 16);
    matrixMul<<<mm_grid, mm_block>>>(d_A, d_B, d_C, MAT_SIZE, MAT_SIZE, MAT_SIZE);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_C, d_C, MAT_SIZE * MAT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("First 5x5 elements of result matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", h_C[i * MAT_SIZE + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_hidden);
    free(h_output);
    free(h_weights1);
    free(h_weights2);
    free(h_A);
    free(h_B);
    free(h_C);
    
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_weights1));
    gpuErrchk(cudaFree(d_hidden));
    gpuErrchk(cudaFree(d_weights2));
    gpuErrchk(cudaFree(d_output));
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    
    printf("\nDeep learning with CUDA example completed successfully.\n");
    printf("This example demonstrated basic neural network operations using CUDA.\n");
    
    return 0;
}