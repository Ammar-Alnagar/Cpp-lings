#include <stdio.h>
#include <cuda_runtime.h>

// Example kernel demonstrating thread hierarchy and synchronization
__global__ void threadHierarchyExample(float *data, int n) {
    // Shared memory for cooperation within a block
    __shared__ float shared_data[256];
    
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load data to shared memory
    if (global_tid < n) {
        shared_data[local_tid] = data[global_tid];
    } else {
        shared_data[local_tid] = 0.0f;
    }
    
    // Synchronize threads in the block
    __syncthreads();
    
    // Perform computation using shared memory
    if (global_tid < n) {
        // Example: each thread adds its left neighbor's value
        float left_val = (local_tid > 0) ? shared_data[local_tid - 1] : 0.0f;
        data[global_tid] = shared_data[local_tid] + left_val;
    }
}

// Example kernel using warp shuffle operations
__global__ void warpShuffleExample(int *input, int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (tid < n) {
        int value = input[tid];
        
        // Get value from the next thread in the warp
        int next_value = __shfl_down_sync(0xFFFFFFFF, value, 1);
        
        // Sum with next thread's value (last thread in warp uses its own)
        output[tid] = value + (lane_id < 31 ? next_value : 0);
    }
}

// Example kernel demonstrating atomic operations
__global__ void atomicExample(int *counter, int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Atomically increment counter
        int current_count = atomicAdd(counter, 1);
        
        // Store the value at the position it was assigned
        data[current_count] = tid;
    }
}

// Example kernel with occupancy hints
__global__ void __launch_bounds__(256, 4)
optimizedKernel(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Simple computation to demonstrate occupancy
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

// Example kernel demonstrating cooperative groups (if supported)
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void cooperativeGroupsExample(float *data, int n) {
    thread_block block = this_thread_block();
    __shared__ float s_data[256];
    
    int local_tid = block.thread_rank();
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    
    // Load data
    if (global_tid < n) {
        s_data[local_tid] = data[global_tid];
    }
    
    // Synchronize using cooperative group
    block.sync();
    
    // Process data
    if (global_tid < n) {
        data[global_tid] = s_data[local_tid] * 2.0f;
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
    const int bytes = N * sizeof(float);
    
    // Allocate host and device memory
    float *h_data = (float*)malloc(bytes);
    float *d_data;
    gpuErrchk(cudaMalloc(&d_data, bytes));
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Setup execution configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch thread hierarchy example
    threadHierarchyExample<<<gridSize, blockSize>>>(d_data, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host to see changes
    gpuErrchk(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("After thread hierarchy example - first 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");
    
    // Reset data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    gpuErrchk(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Launch warp shuffle example
    int *d_input, *d_output;
    gpuErrchk(cudaMalloc(&d_input, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_output, N * sizeof(int)));
    
    // Prepare integer data for shuffle example
    int *h_input = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }
    gpuErrchk(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    
    warpShuffleExample<<<gridSize, blockSize>>>(d_input, d_output, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy results back
    int *h_output = (int*)malloc(N * sizeof(int));
    gpuErrchk(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("After warp shuffle example - first 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    // Atomic operations example
    int *d_counter, *d_atomic_data;
    gpuErrchk(cudaMalloc(&d_counter, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_atomic_data, N * sizeof(int)));
    
    // Initialize counter to 0
    int zero = 0;
    gpuErrchk(cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    atomicExample<<<gridSize, blockSize>>>(d_counter, d_atomic_data, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Check final counter value
    int final_count;
    gpuErrchk(cudaMemcpy(&final_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Atomic counter final value: %d\n", final_count);
    
    // Launch cooperative groups example (if supported)
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    gpuErrchk(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    cooperativeGroupsExample<<<gridSize, blockSize>>>(d_data, N);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("After cooperative groups example - first 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");
    
    // Cleanup
    free(h_data);
    free(h_input);
    free(h_output);
    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));
    gpuErrchk(cudaFree(d_counter));
    gpuErrchk(cudaFree(d_atomic_data));
    
    printf("Thread programming examples completed successfully.\n");
    
    return 0;
}