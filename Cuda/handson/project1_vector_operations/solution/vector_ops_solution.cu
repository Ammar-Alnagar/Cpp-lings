#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper function for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Vector addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector scaling kernel
__global__ void vectorScale(float *a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        a[idx] *= scalar;
    }
}

// Dot product kernel using reduction
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

// Magnitude calculation kernel
__global__ void vectorMagnitude(float *a, float *result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load squared values into shared memory
    sdata[tid] = (idx < n) ? a[idx] * a[idx] : 0.0f;
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

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_scaled = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    gpuErrchk(cudaMalloc(&d_a, size));
    gpuErrchk(cudaMalloc(&d_b, size));
    gpuErrchk(cudaMalloc(&d_c, size));
    
    // Copy data from host to device
    gpuErrchk(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch vector addition kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host
    gpuErrchk(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    printf("First 10 results of vector addition: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_c[i]);
    }
    printf("\n");
    
    // Test vector scaling
    float scale_factor = 2.5f;
    vectorScale<<<gridSize, blockSize>>>(d_a, scale_factor, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy scaled result back
    gpuErrchk(cudaMemcpy(h_scaled, d_a, size, cudaMemcpyDeviceToHost));
    
    printf("First 10 results of vector scaling: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_scaled[i]);
    }
    printf("\n");
    
    // Calculate dot product
    float *d_partial_dot, *h_partial_dot;
    int num_blocks = gridSize;
    gpuErrchk(cudaMalloc(&d_partial_dot, num_blocks * sizeof(float)));
    h_partial_dot = (float*)malloc(num_blocks * sizeof(float));
    
    vectorDotProduct<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, d_b, d_partial_dot, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_partial_dot, d_partial_dot, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Sum partial results on CPU
    float dot_product_result = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        dot_product_result += h_partial_dot[i];
    }
    printf("Dot product result: %.2f\n", dot_product_result);
    
    // Calculate magnitude
    float *d_partial_mag;
    gpuErrchk(cudaMalloc(&d_partial_mag, num_blocks * sizeof(float)));
    
    vectorMagnitude<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, d_partial_mag, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_partial_dot, d_partial_mag, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Sum partial results and take square root
    float mag_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        mag_sum += h_partial_dot[i];
    }
    float magnitude_result = sqrtf(mag_sum);
    printf("Magnitude result: %.2f\n", magnitude_result);
    
    // Cleanup allocated memory
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_partial_dot));
    gpuErrchk(cudaFree(d_partial_mag));
    
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_scaled);
    free(h_partial_dot);
    
    printf("Project 1 completed successfully!\n");
    
    return 0;
}