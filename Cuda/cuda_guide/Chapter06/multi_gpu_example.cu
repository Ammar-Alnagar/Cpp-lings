#include <stdio.h>
#include <cuda_runtime.h>

// Structure to hold GPU-specific data
typedef struct {
    float *d_data;
    cudaStream_t stream;
    int deviceId;
} GpuContext;

// Kernel for processing data
__global__ void processDataKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Kernel for initialization
__global__ void initializeDataKernel(float *data, int n, int gpuId) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.0f + gpuId * 1000.0f;
    }
}

// Local reduction kernel
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
    }
}

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

void multiGpuProcessingExample() {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    
    printf("\n=== Multi-GPU Processing Example ===\n");
    printf("Available GPUs: %d\n", numGpus);
    
    if (numGpus < 1) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    const int totalSize = 8192;
    const int sizePerGpu = totalSize / numGpus;
    if (sizePerGpu == 0) {
        printf("Array too small for number of GPUs, using 1 element per GPU\n");
    }
    
    // Allocate host memory
    float *h_data = (float*)malloc(totalSize * sizeof(float));
    
    // Initialize host data
    for (int i = 0; i < totalSize; i++) {
        h_data[i] = i * 1.0f;
    }
    
    // Initialize GPU contexts
    GpuContext *contexts = (GpuContext*)malloc(numGpus * sizeof(GpuContext));
    initializeMultiGpuContext(contexts, numGpus, sizePerGpu > 0 ? sizePerGpu : 1);
    
    // Distribute work across GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        
        // Determine actual size for this GPU (handle uneven division)
        int actualSize = (i == numGpus - 1) ? (totalSize - i * sizePerGpu) : sizePerGpu;
        if (actualSize <= 0) continue;
        
        // Copy data chunk to GPU
        cudaMemcpyAsync(
            contexts[i].d_data, 
            h_data + i * sizePerGpu, 
            actualSize * sizeof(float), 
            cudaMemcpyHostToDevice, 
            contexts[i].stream
        );
        
        // Launch kernel on this GPU
        processDataKernel<<<actualSize / 256, 256, 0, contexts[i].stream>>>(
            contexts[i].d_data, actualSize
        );
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(contexts[i].stream);
    }
    
    printf("Processing completed on all GPUs.\n");
    
    // Copy results back
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        
        int actualSize = (i == numGpus - 1) ? (totalSize - i * sizePerGpu) : sizePerGpu;
        if (actualSize <= 0) continue;
        
        cudaMemcpyAsync(
            h_data + i * sizePerGpu,
            contexts[i].d_data,
            actualSize * sizeof(float),
            cudaMemcpyDeviceToHost,
            contexts[i].stream
        );
        cudaStreamSynchronize(contexts[i].stream);
    }
    
    printf("First 10 values after multi-GPU processing:\n");
    for (int i = 0; i < 10 && i < totalSize; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
        cudaStreamDestroy(contexts[i].stream);
    }
    free(contexts);
    free(h_data);
}

void multiGpuSynchronizationExample() {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    
    printf("\n=== Multi-GPU Synchronization Example ===\n");
    
    if (numGpus < 2) {
        printf("Need at least 2 GPUs for synchronization example\n");
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
                    printf("Enabled peer access: GPU %d -> GPU %d\n", i, j);
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
    
    printf("Synchronization completed across %d GPUs.\n", numGpus);
    
    // Cleanup
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        cudaFree(contexts[i].d_data);
        cudaStreamDestroy(contexts[i].stream);
    }
    free(contexts);
}

void unifiedMemoryMultiGpu() {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    
    printf("\n=== Unified Memory Multi-GPU Example ===\n");
    
    if (numGpus < 2) {
        printf("Need at least 2 GPUs for unified memory example\n");
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
    cudaMemPrefetchAsync(unified_data, N/2 * sizeof(float), 0);
    processDataKernel<<<256, 256>>>(unified_data, N/2);
    
    // Process on GPU 1
    cudaMemPrefetchAsync(unified_data + N/2, N/2 * sizeof(float), 1);
    processDataKernel<<<256, 256>>>(unified_data + N/2, N/2);
    
    // Wait for both to complete
    cudaDeviceSynchronize();
    
    // Prefetch back to CPU
    cudaMemPrefetchAsync(unified_data, N * sizeof(float), cudaCpuDeviceId);
    
    // Verify results
    printf("First 10 values after unified memory processing:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", unified_data[i]);
    }
    printf("\n");
    
    cudaFree(unified_data);
}

int main() {
    printf("=== Multi-GPU Programming Examples ===\n");
    
    // Query available devices
    queryDevices();
    
    // Run multi-GPU processing example
    multiGpuProcessingExample();
    
    // Run synchronization example
    multiGpuSynchronizationExample();
    
    // Run unified memory example
    unifiedMemoryMultiGpu();
    
    printf("\nAll multi-GPU examples completed successfully.\n");
    printf("Note: On systems with fewer than 2 GPUs, some examples will show limited functionality.\n");
    
    return 0;
}