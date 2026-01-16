#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for vector operations
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel for vector squaring
__global__ void vectorSquare(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

// Kernel for initialization
__global__ void vectorInit(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.0f;
    }
}

// Example kernel for streams demonstration
__global__ void computeIntensiveKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // Perform some intensive computation
        for (int i = 0; i < 100; i++) {
            val = val * 1.01f + sinf(val);
        }
        data[idx] = val;
    }
}

// Grid-stride loop kernel
__global__ void gridStrideKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

// Callback function
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Callback executed: stream completed with status %s\n", 
           cudaGetErrorString(status));
    
    int *counter = (int*)userData;
    (*counter)++;
    printf("Callback counter: %d\n", *counter);
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
    
    // Allocate host and device memory
    float *h_data = (float*)malloc(size);
    float *d_data1, *d_data2;
    gpuErrchk(cudaMalloc(&d_data1, size));
    gpuErrchk(cudaMalloc(&d_data2, size));
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_data1, h_data, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_data2, h_data, size, cudaMemcpyHostToDevice));
    
    printf("=== CUDA Streams Example ===\n");
    
    // Create streams
    cudaStream_t stream1, stream2;
    gpuErrchk(cudaStreamCreate(&stream1));
    gpuErrchk(cudaStreamCreate(&stream2));
    
    // Launch kernels in different streams
    computeIntensiveKernel<<<64, 16, 0, stream1>>>(d_data1, N);
    vectorSquare<<<64, 16, 0, stream2>>>(d_data2, N);
    
    // Synchronize streams
    gpuErrchk(cudaStreamSynchronize(stream1));
    gpuErrchk(cudaStreamSynchronize(stream2));
    
    printf("Streams execution completed.\n");
    
    // Event timing example
    printf("\n=== Event Timing Example ===\n");
    
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    
    // Record start event
    gpuErrchk(cudaEventRecord(start));
    
    // Launch kernel
    vectorSquare<<<64, 16>>>(d_data1, N);
    
    // Record stop event
    gpuErrchk(cudaEventRecord(stop));
    
    // Wait for completion
    gpuErrchk(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Kernel execution time: %.2f ms\n", milliseconds);
    
    // Event synchronization example
    printf("\n=== Event Synchronization Example ===\n");
    
    cudaEvent_t syncEvent;
    gpuErrchk(cudaEventCreate(&syncEvent));
    
    // Launch first kernel
    vectorInit<<<64, 16, 0, stream1>>>(d_data1, N);
    
    // Record event after first kernel
    gpuErrchk(cudaEventRecord(syncEvent, stream1));
    
    // Wait for event in second stream
    gpuErrchk(cudaStreamWaitEvent(stream2, syncEvent, 0));
    
    // Launch second kernel after first completes
    vectorSquare<<<64, 16, 0, stream2>>>(d_data2, N);
    
    gpuErrchk(cudaStreamSynchronize(stream1));
    gpuErrchk(cudaStreamSynchronize(stream2));
    
    printf("Event synchronization completed.\n");
    
    // Grid-stride loop example
    printf("\n=== Grid-Stride Loop Example ===\n");
    
    const int largeN = 10000;  // Larger than typical grid
    const int largeSize = largeN * sizeof(float);
    
    float *d_largeData;
    gpuErrchk(cudaMalloc(&d_largeData, largeSize));
    
    // Initialize large array
    for (int i = 0; i < largeN; i++) {
        h_data = (float*)realloc(h_data, largeSize);
        h_data[i] = i * 0.1f;
    }
    gpuErrchk(cudaMemcpy(d_largeData, h_data, largeSize, cudaMemcpyHostToDevice));
    
    // Use smaller grid with grid-stride loop
    int blockSize = 128;
    int gridSize = 8;  // Much smaller than largeN
    
    gridStrideKernel<<<gridSize, blockSize>>>(d_largeData, largeN);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back
    gpuErrchk(cudaMemcpy(h_data, d_largeData, largeSize, cudaMemcpyDeviceToHost));
    
    printf("First 10 values after grid-stride processing:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");
    
    // Callback example
    printf("\n=== Callback Example ===\n");
    
    int callbackCounter = 0;
    
    // Launch kernel
    vectorSquare<<<64, 16, 0, stream1>>>(d_data1, N);
    
    // Add callback
    gpuErrchk(cudaStreamAddCallback(stream1, myCallback, &callbackCounter, 0));
    
    // Wait for completion
    gpuErrchk(cudaStreamSynchronize(stream1));
    
    printf("Final callback counter: %d\n", callbackCounter);
    
    // Stream priority example
    printf("\n=== Stream Priority Example ===\n");
    
    int leastPriority, greatestPriority;
    gpuErrchk(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    
    printf("Stream priority range: %d to %d\n", leastPriority, greatestPriority);
    
    cudaStream_t highPriorityStream, lowPriorityStream;
    gpuErrchk(cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority));
    gpuErrchk(cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, leastPriority));
    
    // Launch kernels with different priorities
    computeIntensiveKernel<<<32, 16, 0, highPriorityStream>>>(d_data1, N);
    vectorSquare<<<32, 16, 0, lowPriorityStream>>>(d_data2, N);
    
    gpuErrchk(cudaStreamSynchronize(highPriorityStream));
    gpuErrchk(cudaStreamSynchronize(lowPriorityStream));
    
    printf("Priority streams execution completed.\n");
    
    // Cleanup
    gpuErrchk(cudaStreamDestroy(stream1));
    gpuErrchk(cudaStreamDestroy(stream2));
    gpuErrchk(cudaStreamDestroy(highPriorityStream));
    gpuErrchk(cudaStreamDestroy(lowPriorityStream));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    gpuErrchk(cudaEventDestroy(syncEvent));
    gpuErrchk(cudaFree(d_data1));
    gpuErrchk(cudaFree(d_data2));
    gpuErrchk(cudaFree(d_largeData));
    free(h_data);
    
    printf("\nAll kernel execution examples completed successfully.\n");
    
    return 0;
}