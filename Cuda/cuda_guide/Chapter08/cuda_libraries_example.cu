#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <curand.h>
#include <cuda_fp16.h>

// CUDA kernel for basic vector operations
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for vector multiplication
__global__ void vectorMult(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Setup kernel for cuRAND
__global__ void setup_kernel(curandState *state, unsigned long seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Generate random numbers kernel
__global__ void generate_kernel(curandState *state, float *result, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        result[id] = curand_uniform(&state[id]);  // Generate uniform [0,1) random numbers
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
    const int size = N * sizeof(float);
    
    printf("=== CUDA Libraries Example ===\n");
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
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
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    printf("Memory allocated and initialized.\n");
    
    // Basic CUDA kernel example
    printf("\n=== Basic CUDA Kernel Example ===\n");
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    printf("First 10 results of vector addition: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_c[i]);
    }
    printf("\n");
    
    // cuBLAS example
    printf("\n=== cuBLAS Example (SAXPY) ===\n");
    
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    // Restore original values
    gpuErrchk(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Perform SAXPY: y = alpha*x + y
    const float alpha = 2.0f;
    stat = cublasSaxpy(handle, N, &alpha, d_a, 1, d_b, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS SAXPY failed\n");
        return EXIT_FAILURE;
    }
    
    gpuErrchk(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));
    
    printf("First 10 results of SAXPY: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_b[i]);
    }
    printf("\n");
    
    // cuFFT example
    printf("\n=== cuFFT Example ===\n");
    
    const int FFT_N = 256;
    const int fft_size = FFT_N * 2 * sizeof(float); // Complex numbers (real, imag)
    
    float *h_signal = (float*)malloc(fft_size);
    float *d_signal;
    gpuErrchk(cudaMalloc(&d_signal, fft_size));
    
    // Initialize signal (real part = sine wave, imaginary part = 0)
    for (int i = 0; i < FFT_N; i++) {
        h_signal[2*i] = sinf(2.0f * M_PI * i / FFT_N);  // Real part
        h_signal[2*i + 1] = 0.0f;                        // Imaginary part
    }
    
    gpuErrchk(cudaMemcpy(d_signal, h_signal, fft_size, cudaMemcpyHostToDevice));
    
    // Create FFT plan
    cufftHandle plan;
    cufftResult fft_result = cufftPlan1d(&plan, FFT_N, CUFFT_C2C, 1);  // Complex-to-complex 1D FFT
    if (fft_result != CUFFT_SUCCESS) {
        printf("cuFFT plan creation failed\n");
        return EXIT_FAILURE;
    }
    
    // Execute FFT
    fft_result = cufftExecC2C(plan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD);
    if (fft_result != CUFFT_SUCCESS) {
        printf("cuFFT execution failed\n");
        return EXIT_FAILURE;
    }
    
    gpuErrchk(cudaMemcpy(h_signal, d_signal, fft_size, cudaMemcpyDeviceToHost));
    
    printf("First 5 FFT results (real, imag): ");
    for (int i = 0; i < 5; i++) {
        printf("(%.2f, %.2f) ", h_signal[2*i], h_signal[2*i + 1]);
    }
    printf("\n");
    
    // cuRAND example
    printf("\n=== cuRAND Example ===\n");
    
    curandState *d_state;
    float *d_random, *h_random;
    
    gpuErrchk(cudaMalloc(&d_state, N * sizeof(curandState)));
    gpuErrchk(cudaMalloc(&d_random, N * sizeof(float)));
    h_random = (float*)malloc(N * sizeof(float));
    
    // Setup random states
    setup_kernel<<<(N + 255) / 256, 256>>>(d_state, time(NULL), N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Generate random numbers
    generate_kernel<<<(N + 255) / 256, 256>>>(d_state, d_random, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy results back
    gpuErrchk(cudaMemcpy(h_random, d_random, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("First 10 random numbers: ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", h_random[i]);
    }
    printf("\n");
    
    // Half precision example
    printf("\n=== Half Precision Example ===\n");
    
    const int HP_N = 128;
    __half *h_hp_input = (__half*)malloc(HP_N * sizeof(__half));
    __half *h_hp_output = (__half*)malloc(HP_N * sizeof(__half));
    __half *d_hp_input, *d_hp_output;
    
    gpuErrchk(cudaMalloc(&d_hp_input, HP_N * sizeof(__half)));
    gpuErrchk(cudaMalloc(&d_hp_output, HP_N * sizeof(__half)));
    
    // Initialize with half precision values
    for (int i = 0; i < HP_N; i++) {
        h_hp_input[i] = __float2half(i * 0.1f);
    }
    
    gpuErrchk(cudaMemcpy(d_hp_input, h_hp_input, HP_N * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Simple kernel to demonstrate half precision
    for (int i = 0; i < HP_N; i++) {
        h_hp_output[i] = __float2half(__half2float(h_hp_input[i]) * 2.0f);
    }
    
    gpuErrchk(cudaMemcpy(d_hp_output, h_hp_output, HP_N * sizeof(__half), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_hp_output, d_hp_output, HP_N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    printf("First 10 half-precision results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", __half2float(h_hp_output[i]));
    }
    printf("\n");
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_signal);
    free(h_random);
    free(h_hp_input);
    free(h_hp_output);
    
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_signal));
    gpuErrchk(cudaFree(d_random));
    gpuErrchk(cudaFree(d_state));
    gpuErrchk(cudaFree(d_hp_input));
    gpuErrchk(cudaFree(d_hp_output));
    
    // Cleanup libraries
    cublasDestroy(handle);
    cufftDestroy(plan);
    
    printf("\nAll CUDA libraries examples completed successfully.\n");
    
    return 0;
}