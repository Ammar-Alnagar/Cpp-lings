# Chapter 6: Multi-GPU Programming

## Overview
Multi-GPU programming allows you to harness the power of multiple GPUs to solve larger problems or achieve higher performance. This chapter covers techniques for distributing work across multiple GPUs, managing data movement between devices, and synchronizing operations across multiple devices.

## GPU Device Management

### Querying Available Devices
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

void queryDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max block dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
```

### Setting and Checking Current Device
```cuda
void setDeviceExample() {
    int currentDevice;
    cudaGetDevice(&currentDevice);
    printf("Currently using device %d\n", currentDevice);
    
    // Switch to device 1 (if available)
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount > 1) {
        cudaSetDevice(1);
        cudaGetDevice(&currentDevice);
        printf("Switched to device %d\n", currentDevice);
    }
}
```

## Peer-to-Peer (P2P) Memory Access

### Enabling and Using P2P Access
```cuda
bool enablePeerAccess(int srcDevice, int dstDevice) {
    int canAccessPeer;
    cudaDeviceCanAccessPeer(&canAccessPeer, srcDevice, dstDevice);
    
    if (canAccessPeer) {
        cudaSetDevice(srcDevice);
        cudaDeviceEnablePeerAccess(dstDevice, 0);
        printf("Enabled peer access from device %d to device %d\n", srcDevice, dstDevice);
        return true;
    } else {
        printf("Peer access not possible between device %d and %d\n", srcDevice, dstDevice);
        return false;
    }
}

void p2pMemoryCopy(float *src, float *dst, size_t size, int srcDevice, int dstDevice) {
    if (enablePeerAccess(srcDevice, dstDevice)) {
        cudaSetDevice(srcDevice);
        // Direct memory copy between GPUs without staging through host
        cudaMemcpyPeer(dst, dstDevice, src, srcDevice, size);
    }
}
```

## Multi-GPU Data Distribution

### Splitting Work Across Multiple GPUs
```cuda
// Structure to hold GPU-specific data
typedef struct {
    float *d_data;
    cudaStream_t stream;
    int deviceId;
} GpuContext;

void initializeMultiGpuContext(GpuContext *contexts, int numGpus, int dataSizePerGpu) {
    for (int i = 0; i < numGpus; i++) {
        contexts[i].deviceId = i;
        
        // Set device
        cudaSetDevice(i);
        
        // Allocate device memory
        cudaMalloc(&contexts[i].d_data, dataSizePerGpu * sizeof(float));
        
        // Create stream
        cudaStreamCreate(&contexts[i].stream);
    }
}

void distributeWorkMultiGpu(float *h_data, int totalSize, int numGpus) {
    int sizePerGpu = totalSize / numGpus;
    
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    initializeMultiGpuContext(contexts, numGpus, sizePerGpu);
    
    // Distribute data to each GPU
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        
        // Copy data chunk to GPU
        cudaMemcpyAsync(
            contexts[i].d_data, 
            h_data + i * sizePerGpu, 
            sizePerGpu * sizeof(float), 
            cudaMemcpyHostToDevice, 
            contexts[i].stream
        );
        
        // Launch kernel on this GPU
        processDataKernel<<<sizePerGpu / 256, 256, 0, contexts[i].stream>>>(
            contexts[i].d_data, sizePerGpu
        );
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(contexts[i].stream);
    }
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
        cudaStreamDestroy(contexts[i].stream);
    }
    free(contexts);
}

__global__ void processDataKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
```

## Multi-GPU Synchronization

### Synchronizing Across Multiple GPUs
```cuda
void multiGpuSynchronizationExample() {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    
    if (numGpus < 2) {
        printf("Need at least 2 GPUs for this example\n");
        return;
    }
    
    // Enable peer access between GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < numGpus; j++) {
            if (i != j) {
                int canAccessPeer;
                cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }
    
    // Create contexts for each GPU
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    for (int i = 0; i < numGpus; i++) {
        contexts[i].deviceId = i;
        cudaSetDevice(i);
        cudaMalloc(&contexts[i].d_data, 1024 * sizeof(float));
        cudaStreamCreate(&contexts[i].stream);
    }
    
    // Launch work on all GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        initializeDataKernel<<<32, 32, 0, contexts[i].stream>>>(
            contexts[i].d_data, 1024, i
        );
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(contexts[i].stream);
    }
    
    // Perform cross-GPU operations if needed
    if (numGpus >= 2) {
        cudaSetDevice(0);
        // Copy data from GPU 1 to GPU 0
        cudaMemcpyPeerAsync(
            contexts[0].d_data, 0,
            contexts[1].d_data, 1,
            1024 * sizeof(float),
            contexts[0].stream
        );
        cudaStreamSynchronize(contexts[0].stream);
    }
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
        cudaStreamDestroy(contexts[i].stream);
    }
    free(contexts);
}

__global__ void initializeDataKernel(float *data, int n, int gpuId) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.0f + gpuId * 1000.0f;
    }
}
```

## Unified Memory with Multiple GPUs

### Managing Unified Memory Across Multiple GPUs
```cuda
void unifiedMemoryMultiGpu() {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    
    if (numGpus < 2) {
        printf("Need at least 2 GPUs for this example\n");
        return;
    }
    
    const int N = 1024 * 1024;
    float *unified_data;
    
    // Allocate unified memory
    cudaMallocManaged(&unified_data, N * sizeof(float));
    
    // Initialize on CPU
    for (int i = 0; i < N; i++) {
        unified_data[i] = i * 1.0f;
    }
    
    // Process on GPU 0
    cudaMemPrefetchAsync(unified_data, N * sizeof(float), 0);
    processOnGpu0<<<256, 256>>>(unified_data, N/2);
    
    // Process on GPU 1
    cudaMemPrefetchAsync(unified_data + N/2, N/2 * sizeof(float), 1);
    processOnGpu1<<<256, 256>>>(unified_data + N/2, N/2);
    
    // Wait for both to complete
    cudaDeviceSynchronize();
    
    // Prefetch back to CPU
    cudaMemPrefetchAsync(unified_data, N * sizeof(float), cudaCpuDeviceId);
    
    // Verify results
    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", unified_data[i]);
    }
    printf("\n");
    
    cudaFree(unified_data);
}

__global__ void processOnGpu0(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void processOnGpu1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}
```

## Multi-GPU Collective Operations

### Implementing All-Reduce Across Multiple GPUs
```cuda
void multiGpuAllReduce(float *input, float *output, int n, int numGpus) {
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    
    // Distribute input data across GPUs
    int elementsPerGpu = n / numGpus;
    
    for (int i = 0; i < numGpus; i++) {
        contexts[i].deviceId = i;
        cudaSetDevice(i);
        
        cudaMalloc(&contexts[i].d_data, elementsPerGpu * sizeof(float));
        
        // Copy portion of input to this GPU
        cudaMemcpy(
            contexts[i].d_data,
            input + i * elementsPerGpu,
            elementsPerGpu * sizeof(float),
            cudaMemcpyHostToDevice
        );
    }
    
    // Perform local reduction on each GPU
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        localReduceKernel<<<1, 256>>>(
            contexts[i].d_data, elementsPerGpu
        );
    }
    
    // Collect partial results and perform final reduction
    float *partialResults = (float*)malloc(numGpus * sizeof(float));
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaMemcpy(
            &partialResults[i],
            contexts[i].d_data,
            sizeof(float),
            cudaMemcpyDeviceToHost
        );
    }
    
    // Compute final sum on CPU
    float finalSum = 0.0f;
    for (int i = 0; i < numGpus; i++) {
        finalSum += partialResults[i];
    }
    
    // Broadcast result back to all GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaMemcpy(
            contexts[i].d_data,
            &finalSum,
            sizeof(float),
            cudaMemcpyHostToDevice
        );
        
        // Copy result back to output
        cudaMemcpy(
            output + i * elementsPerGpu,
            contexts[i].d_data,
            sizeof(float),
            cudaMemcpyDeviceToHost
        );
    }
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
    }
    free(contexts);
    free(partialResults);
}

__global__ void localReduceKernel(float *data, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result back to global memory
    if (tid == 0) {
        data[0] = sdata[0];
    }
}
```

## NCCL for Multi-GPU Communication

### Using NCCL for Collective Communications
```cuda
#ifdef USE_NCCL
#include <nccl.h>

void ncclAllReduceExample(float *sendbuff, float *recvbuff, int count, int nDevices) {
    ncclComm_t *comms = (ncclComm_t*)malloc(nDevices * sizeof(ncclComm_t));
    cudaStream_t *streams = (cudaStream_t*)malloc(nDevices * sizeof(cudaStream_t));
    
    // Initialize NCCL communicators
    int *devList = (int*)malloc(nDevices * sizeof(int));
    for (int i = 0; i < nDevices; i++) {
        devList[i] = i;
    }
    
    ncclCommInitAll(comms, nDevices, devList);
    
    // Create streams
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }
    
    // Perform all-reduce operation
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        ncclAllReduce(
            sendbuff + i * count,  // send buffer for this GPU
            recvbuff + i * count,  // receive buffer for this GPU
            count,                 // number of elements
            ncclFloat,             // data type
            ncclSum,               // reduction operation
            comms[i],              // communicator
            streams[i]             // CUDA stream
        );
    }
    
    // Synchronize
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }
    ncclGroupEnd();
    for (int i = 0; i < nDevices; i++) {
        ncclCommDestroy(comms[i]);
    }
    
    free(comms);
    free(streams);
    free(devList);
}
#endif
```

## Multi-GPU Performance Considerations

### Optimizing Data Movement
```cuda
void optimizedMultiGpuProcessing(float *h_data, int totalSize, int numGpus) {
    int sizePerGpu = totalSize / numGpus;
    
    // Pre-allocate all resources
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    
    for (int i = 0; i < numGpus; i++) {
        contexts[i].deviceId = i;
        cudaSetDevice(i);
        
        // Use pinned memory for faster transfers
        float *h_pinned;
        cudaMallocHost(&h_pinned, sizePerGpu * sizeof(float));
        
        // Async memory allocation
        cudaMalloc(&contexts[i].d_data, sizePerGpu * sizeof(float));
        cudaStreamCreate(&contexts[i].stream);
        
        // Async memory copy
        cudaMemcpyAsync(
            contexts[i].d_data,
            h_data + i * sizePerGpu,
            sizePerGpu * sizeof(float),
            cudaMemcpyHostToDevice,
            contexts[i].stream
        );
        
        // Launch kernel asynchronously
        processKernelAsync<<<64, 256, 0, contexts[i].stream>>>(
            contexts[i].d_data, sizePerGpu
        );
        
        // Copy result back asynchronously
        cudaMemcpyAsync(
            h_pinned,
            contexts[i].d_data,
            sizePerGpu * sizeof(float),
            cudaMemcpyDeviceToHost,
            contexts[i].stream
        );
    }
    
    // Wait for all operations to complete
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(contexts[i].stream);
    }
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
        cudaStreamDestroy(contexts[i].stream);
    }
    free(contexts);
}

__global__ void processKernelAsync(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // More complex processing
        float val = data[idx];
        val = val * val + sinf(val) + cosf(val);
        data[idx] = val;
    }
}
```

## Error Handling in Multi-GPU Applications

### Multi-GPU Error Checking
```cuda
#define gpuErrchkMulti(ans, deviceId) { gpuAssertMulti((ans), (deviceId), __FILE__, __LINE__); }

inline void gpuAssertMulti(cudaError_t code, int deviceId, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU %d: %s %s %d\n", deviceId, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void safeMultiGpuOperation(float *h_data, int totalSize, int numGpus) {
    int sizePerGpu = totalSize / numGpus;
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    
    for (int i = 0; i < numGpus; i++) {
        contexts[i].deviceId = i;
        cudaSetDevice(i);
        
        // Safe allocation with error checking
        cudaError_t err = cudaMalloc(&contexts[i].d_data, sizePerGpu * sizeof(float));
        gpuErrchkMulti(err, i);
        
        if (err != cudaSuccess) {
            // Handle error appropriately
            printf("Failed to allocate memory on GPU %d\n", i);
            continue;
        }
        
        cudaStreamCreate(&contexts[i].stream);
        
        // Safe memory copy
        err = cudaMemcpyAsync(
            contexts[i].d_data,
            h_data + i * sizePerGpu,
            sizePerGpu * sizeof(float),
            cudaMemcpyHostToDevice,
            contexts[i].stream
        );
        gpuErrchkMulti(err, i);
        
        // Safe kernel launch
        processKernelAsync<<<64, 256, 0, contexts[i].stream>>>(
            contexts[i].d_data, sizePerGpu
        );
        
        // Check for launch errors
        err = cudaGetLastError();
        gpuErrchkMulti(err, i);
    }
    
    // Synchronize and check for execution errors
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaError_t err = cudaStreamSynchronize(contexts[i].stream);
        gpuErrchkMulti(err, i);
    }
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        if (contexts[i].d_data) {
            cudaSetDevice(i);
            cudaFree(contexts[i].d_data);
            cudaStreamDestroy(contexts[i].stream);
        }
    }
    free(contexts);
}
```

## Multi-GPU Application Patterns

### Pipeline Processing Across Multiple GPUs
```cuda
void pipelineMultiGpuProcessing(float *input, float *output, int totalSize, int numGpus) {
    int chunkSize = totalSize / numGpus;
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    
    // Initialize all GPU contexts
    for (int i = 0; i < numGpus; i++) {
        contexts[i].deviceId = i;
        cudaSetDevice(i);
        cudaMalloc(&contexts[i].d_data, chunkSize * sizeof(float));
        cudaStreamCreate(&contexts[i].stream);
    }
    
    // Pipeline: each GPU processes a different stage of the same data
    for (int iteration = 0; iteration < numGpus; iteration++) {
        for (int gpu = 0; gpu < numGpus; gpu++) {
            int dataChunk = (iteration + gpu) % numGpus;
            
            if (iteration * numGpus + dataChunk < totalSize / chunkSize) {
                cudaSetDevice(gpu);
                
                // Stage 1: Load data
                cudaMemcpyAsync(
                    contexts[gpu].d_data,
                    input + dataChunk * chunkSize,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice,
                    contexts[gpu].stream
                );
                
                // Stage 2: Process data (different processing on each GPU)
                pipelineKernel<<<chunkSize / 256, 256, 0, contexts[gpu].stream>>>(
                    contexts[gpu].d_data, chunkSize, gpu
                );
                
                // Stage 3: Store result
                cudaMemcpyAsync(
                    output + dataChunk * chunkSize,
                    contexts[gpu].d_data,
                    chunkSize * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    contexts[gpu].stream
                );
            }
        }
    }
    
    // Wait for all operations to complete
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(contexts[i].stream);
    }
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
        cudaStreamDestroy(contexts[i].stream);
    }
    free(contexts);
}

__global__ void pipelineKernel(float *data, int n, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        switch(stage % 3) {
            case 0: // Stage 1: Multiply
                data[idx] *= 2.0f;
                break;
            case 1: // Stage 2: Add
                data[idx] += 1.0f;
                break;
            case 2: // Stage 3: Square
                data[idx] = data[idx] * data[idx];
                break;
        }
    }
}
```

## Summary
This chapter covered essential multi-GPU programming techniques:
- Device management and querying
- Peer-to-peer memory access
- Data distribution strategies
- Multi-GPU synchronization methods
- Unified memory across multiple GPUs
- Collective operations (all-reduce example)
- NCCL for optimized collective communications
- Performance optimization techniques
- Error handling in multi-GPU applications
- Application patterns like pipelining

Multi-GPU programming allows scaling applications across multiple devices to handle larger datasets and achieve higher performance. The next chapter will explore parallel programming patterns in CUDA.