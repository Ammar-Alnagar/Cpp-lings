# Chapter 4: Kernel Execution in CUDA

## Overview
Kernel execution is a critical aspect of CUDA programming that involves launching functions on the GPU, managing execution contexts, and optimizing performance. This chapter covers various aspects of kernel execution including streams, events, callbacks, and advanced execution features.

## CUDA Streams

### Introduction to Streams
CUDA streams allow for asynchronous execution of kernels and memory operations, enabling overlapping of computation and data transfer.

```cuda
#include <cuda_runtime.h>

// Example of using streams for concurrent execution
void streamExample() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // Allocate host and device memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *d_a, *d_b;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    
    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Asynchronous memory copies in different streams
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);
    
    // Launch kernels in different streams
    vectorAdd<<<128, 8, 0, stream1>>>(d_a, d_a, d_a, N);
    vectorAdd<<<128, 8, 0, stream2>>>(d_b, d_b, d_b, N);
    
    // Copy results back asynchronously
    cudaMemcpyAsync(h_a, d_a, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_b, d_b, size, cudaMemcpyDeviceToHost, stream2);
    
    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
}
```

### Default Stream vs Concurrent Streams
```cuda
// Default stream (stream 0) - synchronous behavior
__global__ void kernel1(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}

// Custom streams - asynchronous behavior
__global__ void kernel2(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] += 1.0f;
}

void compareDefaultVsCustomStreams() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Using default stream - operations execute sequentially
    kernel1<<<32, 32>>>(d_data);  // Executes first
    kernel2<<<32, 32>>>(d_data);  // Executes after kernel1 completes
    
    // Using custom streams - operations can overlap
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    kernel1<<<32, 32, 0, stream1>>>(d_data);  // May overlap with kernel2
    kernel2<<<32, 32, 0, stream2>>>(d_data);  // May overlap with kernel1
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);
}
```

## CUDA Events

### Timing Kernel Execution
CUDA events are used to measure execution time and synchronize operations.

```cuda
void eventTimingExample() {
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel
    vectorSquare<<<256, 256>>>(d_data, N);
    
    // Record stop event
    cudaEventRecord(stop);
    
    // Wait for completion
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %.2f ms\n", milliseconds);
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
}

__global__ void vectorSquare(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}
```

### Event-Based Synchronization
```cuda
void eventSynchronizationExample() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, size);
    cudaMalloc(&d_data2, size);
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Launch first kernel in stream1
    vectorInit<<<64, 64, 0, stream1>>>(d_data1, N);
    
    // Create event and record it in stream1
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream1);
    
    // Wait for event in stream2 (creates dependency)
    cudaStreamWaitEvent(stream2, event, 0);
    
    // Launch second kernel in stream2 (will wait for first to complete)
    vectorCopy<<<64, 64, 0, stream2>>>(d_data1, d_data2, N);
    
    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Clean up
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);
}

__global__ void vectorInit(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.0f;
    }
}

__global__ void vectorCopy(float *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx] * 2.0f;
    }
}
```

## Stream Priorities

### Using Priority Streams
CUDA supports priority streams for better resource management.

```cuda
void priorityStreamExample() {
    // Query device for stream priority range
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    printf("Stream priority range: %d to %d\n", leastPriority, greatestPriority);
    
    // Create streams with different priorities
    cudaStream_t highPriorityStream, lowPriorityStream;
    cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority);
    cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, leastPriority);
    
    // Launch kernels with different priorities
    // High priority kernels get scheduling preference
    computeIntensiveKernel<<<256, 256, 0, highPriorityStream>>>(/* args */);
    lessCriticalKernel<<<256, 256, 0, lowPriorityStream>>>(/* args */);
    
    // Clean up
    cudaStreamDestroy(highPriorityStream);
    cudaStreamDestroy(lowPriorityStream);
}

__global__ void computeIntensiveKernel(/* parameters */) {
    // Resource-intensive computation
    for (int i = 0; i < 1000; i++) {
        // Some computation
    }
}

__global__ void lessCriticalKernel(/* parameters */) {
    // Less critical computation
}
```

## CUDA Callbacks

### Using Asynchronous Callbacks
Callbacks allow host functions to be executed when GPU operations complete.

```cuda
// Callback function
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Callback executed: stream completed with status %s\n", 
           cudaGetErrorString(status));
    
    // Perform host-side operations after GPU work completes
    int *counter = (int*)userData;
    (*counter)++;
    printf("Callback counter: %d\n", *counter);
}

void callbackExample() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int callbackCounter = 0;
    
    // Launch kernel
    vectorOperation<<<64, 64, 0, stream>>>(d_data, N);
    
    // Add callback to be executed when preceding operations complete
    cudaStreamAddCallback(stream, myCallback, &callbackCounter, 0);
    
    // Wait for everything to complete
    cudaStreamSynchronize(stream);
    
    printf("Final callback counter: %d\n", callbackCounter);
    
    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_data);
}

__global__ void vectorOperation(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 3.0f + 1.0f;
    }
}
```

## Dynamic Parallelism

### Kernels Launching Other Kernels
Dynamic parallelism allows kernels to launch child kernels.

```cuda
// Child kernel
__global__ void childKernel(float *data, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[offset + idx] *= 2.0f;
    }
}

// Parent kernel that launches child kernels
__global__ void parentKernel(float *data, int n) {
    int parentId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (parentId == 0) {  // Only one thread launches children
        // Configure child kernel launch parameters
        dim3 childGrid(2);
        dim3 childBlock(128);
        
        // Launch child kernels from device code
        childKernel<<<childGrid, childBlock>>>(data, 0, 128);
        childKernel<<<childGrid, childBlock>>>(data, 256, 128);
        
        // Synchronize child kernels
        cudaDeviceSynchronize();
    }
}

void dynamicParallelismExample() {
    const int N = 512;
    const int size = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Initialize data
    initializeData<<<16, 32>>>(d_data, N);
    cudaDeviceSynchronize();
    
    // Launch parent kernel that will spawn children
    parentKernel<<<1, 1>>>(d_data, N);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
}

__global__ void initializeData(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.0f;
    }
}
```

## Grid-Stride Loops

### Efficient Processing of Large Arrays
Grid-stride loops allow kernels to process arrays larger than the grid size.

```cuda
__global__ void gridStrideKernel(float *data, int n) {
    // Calculate initial index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process elements with stride
    for (int i = idx; i < n; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

void gridStrideExample() {
    const int N = 1000000;  // Much larger than typical grid size
    const int size = N * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.1f;
    }
    
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Launch with smaller grid than data size
    int blockSize = 256;
    int gridSize = 32;  // Much smaller than N
    
    gridStrideKernel<<<gridSize, blockSize>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("First 10 processed values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");
    
    free(h_data);
    cudaFree(d_data);
}
```

## Occupancy and Launch Bounds

### Optimizing Kernel Launch Configuration
Using launch bounds to optimize register usage and occupancy.

```cuda
// Specify launch bounds to optimize for occupancy
__global__ void __launch_bounds__(256, 4)
optimizedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Computation that benefits from the specified bounds
        float val = data[idx];
        val = val * val + sinf(val) + cosf(val);
        data[idx] = val;
    }
}

// Alternative: optimize for minimum memory per thread
__global__ void __launch_bounds__(512, 2)
memoryEfficientKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simpler computation to reduce register pressure
        data[idx] = data[idx] * 2.0f;
    }
}
```

## Error Handling in Kernel Execution

### Proper Error Checking
```cuda
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void safeKernelExecution() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    float *d_data;
    gpuErrchk(cudaMalloc(&d_data, size));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch kernel with error checking
    vectorProcess<<<gridSize, blockSize>>>(d_data, N);
    
    // Check for launch errors
    gpuErrchk(cudaGetLastError());
    
    // Wait for completion and check for execution errors
    gpuErrchk(cudaDeviceSynchronize());
    
    cudaFree(d_data);
}

__global__ void vectorProcess(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
```

## Performance Considerations

### Measuring and Optimizing Kernel Launch Overhead
```cuda
void performanceBenchmark() {
    const int N = 1024 * 1024;
    const int size = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Time multiple small kernel launches
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numKernels = 1000;
    
    cudaEventRecord(start);
    for (int i = 0; i < numKernels; i++) {
        smallKernel<<<1, 32>>>(d_data, i);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float smallKernelTime;
    cudaEventElapsedTime(&smallKernelTime, start, stop);
    
    // Time single large kernel
    cudaEventRecord(start);
    largeKernel<<<32, 256>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float largeKernelTime;
    cudaEventElapsedTime(&largeKernelTime, start, stop);
    
    printf("Time for %d small kernels: %.2f ms\n", numKernels, smallKernelTime);
    printf("Time for 1 large kernel: %.2f ms\n", largeKernelTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
}

__global__ void smallKernel(float *data, int offset) {
    if (threadIdx.x == 0) {
        data[offset] = offset * 1.0f;
    }
}

__global__ void largeKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.0f;
    }
}
```

## Summary
This chapter covered essential aspects of kernel execution in CUDA:
- CUDA streams for asynchronous execution
- Events for timing and synchronization
- Stream priorities for resource management
- Callbacks for host-side notifications
- Dynamic parallelism for kernels launching other kernels
- Grid-stride loops for processing large datasets
- Launch bounds for optimization
- Error handling best practices
- Performance considerations for kernel launches

Understanding these concepts is crucial for building efficient and well-structured CUDA applications. The next chapter will focus on debugging and profiling CUDA applications.