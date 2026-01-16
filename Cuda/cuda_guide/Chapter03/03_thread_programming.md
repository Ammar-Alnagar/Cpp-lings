# Chapter 3: Thread Programming in CUDA

## Overview
Thread programming is fundamental to CUDA. Understanding how threads are organized, scheduled, and synchronized is crucial for writing efficient parallel algorithms. This chapter explores the CUDA thread hierarchy, synchronization mechanisms, and advanced threading concepts.

## Thread Hierarchy

### Warp Execution
A warp is a group of 32 consecutive threads that execute in lockstep. All threads in a warp execute the same instruction at the same time (SIMT - Single Instruction, Multiple Thread).

```cuda
__global__ void warpExecutionExample(int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // All threads in a warp execute this together
    if (tid < n) {
        data[tid] = tid * 2;
    }
    
    // Check if we're the first thread in the warp
    if ((threadIdx.x % 32) == 0) {
        printf("Warp leader thread %d in block %d\n", threadIdx.x, blockIdx.x);
    }
}
```

### Block and Grid Organization
Threads are organized into blocks, and blocks are organized into a grid:

```cuda
// 1D organization
dim3 blockSize(256);           // 256 threads per block
dim3 gridSize((N + 255)/256);  // Enough blocks to cover N elements

// 2D organization
dim3 blockSize(16, 16);        // 16x16 = 256 threads per block
dim3 gridSize((width + 15)/16, (height + 15)/16);

// 3D organization
dim3 blockSize(8, 8, 8);       // 8x8x8 = 512 threads per block
dim3 gridSize((dimX + 7)/8, (dimY + 7)/8, (dimZ + 7)/8);
```

## Thread Synchronization

### Block-Level Synchronization
Threads within a block can synchronize using `__syncthreads()`:

```cuda
__global__ void blockSynchronizationExample(float *data, int n) {
    __shared__ float temp[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < n) {
        temp[local_tid] = data[tid];
    }
    
    // Synchronize all threads in the block
    __syncthreads();
    
    // Now all threads have loaded their data
    // Safe to perform operations that depend on other threads' data
    if (tid < n) {
        // Example: each thread adds its neighbor's value
        if (local_tid > 0) {
            data[tid] += temp[local_tid - 1];
        }
    }
}
```

### Cooperative Groups
CUDA 9.0 introduced cooperative groups for more flexible synchronization:

```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void cooperativeGroupsExample(float *data, int n) {
    // Create thread block group
    thread_block block = this_thread_block();
    
    __shared__ float temp[256];
    int tid = block.thread_rank();
    
    if (block.thread_rank() < n) {
        temp[tid] = data[block.thread_rank()];
    }
    
    // Synchronize using cooperative group
    block.sync();
    
    if (block.thread_rank() < n) {
        data[block.thread_rank()] = temp[tid] * 2.0f;
    }
}
```

## Warp-Level Primitives

### Warp Shuffle Operations
Warp shuffle operations allow threads within a warp to exchange data without using shared memory:

```cuda
__global__ void warpShuffleExample(int *input, int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    if (tid < n) {
        int value = input[tid];
        
        // Shuffle: get value from thread at different lane
        int offsetValue = __shfl_down_sync(0xFFFFFFFF, value, 1);
        
        // Each thread gets the value from the next thread in the warp
        output[tid] = value + offsetValue;
    }
}

// Example: warp-level reduction using shuffle
__device__ float warpReduce(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void warpReduceKernel(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate initial sum for this thread
    float sum = (tid < n) ? input[tid] : 0.0f;
    
    // Reduce within warp
    sum = warpReduce(sum);
    
    // Only the first thread in each warp writes the result
    if ((threadIdx.x % 32) == 0) {
        output[tid / 32] = sum;
    }
}
```

## Occupancy and Performance

### Calculating Occupancy
Occupancy is the ratio of active warps to the maximum number of warps supported by an SM:

```cuda
#include <cuda_runtime.h>

void calculateOccupancy() {
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    // Calculate theoretical occupancy
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                       myKernel, 0, 0);
    
    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, 
                                                 myKernel, blockSize, 0);
    
    float occupancy = (maxActiveBlocks * blockSize / prop.warpSize) / 
                      (float)(prop.maxThreadsPerMultiProcessor / prop.warpSize);
    
    printf("Achieved occupancy: %.1f%%\n", occupancy * 100);
}
```

### Occupancy Optimization Example
```cuda
// Kernel with minimal register usage
__global__ void __launch_bounds__(256, 4) 
optimizedKernel(float *data, int n) {
    // Limit to 256 threads per block, aim for 4 blocks per SM
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Keep register usage low to improve occupancy
        float val = data[tid];
        val = val * val + 1.0f;
        data[tid] = val;
    }
}
```

## Divergent Branching

### Handling Divergence
When threads in a warp take different execution paths, it causes divergence and reduces performance:

```cuda
// PROBLEMATIC: Causes divergence
__global__ void divergentBranching(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        if (input[tid] > 0.0f) {
            // Positive numbers: expensive computation
            output[tid] = sqrt(input[tid]) * 2.0f;
        } else {
            // Negative numbers: different computation
            output[tid] = fabs(input[tid]) * 0.5f;
        }
    }
}

// BETTER: Minimize divergence
__global__ void reducedDivergence(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float val = input[tid];
        float result;
        
        // Separate the computation paths
        if (val > 0.0f) {
            result = sqrt(val) * 2.0f;
        } else {
            result = fabs(val) * 0.5f;
        }
        
        output[tid] = result;
    }
}
```

## Advanced Threading Patterns

### Producer-Consumer Pattern
```cuda
__global__ void producerConsumer(float *input, float *output, int n) {
    __shared__ volatile int buffer_full;
    __shared__ float shared_buffer[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (buffer_full == 0) {
        // Producer phase: fill the buffer
        if (local_tid < 256 && (blockIdx.x * blockDim.x + local_tid) < n) {
            shared_buffer[local_tid] = input[blockIdx.x * blockDim.x + local_tid];
        }
        
        // Signal that buffer is filled
        if (local_tid == 0) {
            buffer_full = 1;
        }
        
        __syncthreads();
        
        // Consumer phase: process the buffer
        if (local_tid < 256 && (blockIdx.x * blockDim.x + local_tid) < n) {
            output[blockIdx.x * blockDim.x + local_tid] = 
                shared_buffer[local_tid] * 2.0f;
        }
    }
}
```

### Reduction with Proper Synchronization
```cuda
__global__ void optimizedReduction(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Perform first level of reduction
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    sdata[tid] += (i + blockDim.x < n) ? input[i + blockDim.x] : 0.0f;
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
        output[blockIdx.x] = sdata[0];
    }
}
```

## Atomic Operations

### Using Atomics for Race Condition Prevention
```cuda
__global__ void atomicOperations(int *histogram, float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Convert floating point value to histogram bin
        int bin = (int)(data[tid] * 10.0f) % 256;
        
        // Atomically increment the histogram bin
        atomicAdd(&histogram[bin], 1);
    }
}

// Different atomic operations
__global__ void atomicExamples(int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Integer atomics
        atomicAdd(&data[0], 1);           // Add
        atomicExch(&data[1], tid);       // Exchange
        atomicMax(&data[2], tid);        // Maximum
        atomicMin(&data[3], tid);        // Minimum
        atomicInc((unsigned int*)&data[4], 10); // Increment with modulus
        atomicCAS(&data[5], 0, tid);     // Compare and swap
        
        // Float atomics (compute capability 2.x and later)
        atomicAdd((float*)&data[6], 1.0f);
    }
}
```

## Thread Divergence Best Practices

### Optimizing Memory Access Patterns
```cuda
// Coalesced access pattern
__global__ void coalescedAccess(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Consecutive threads access consecutive memory locations
    if (tid < n) {
        output[tid] = input[tid] * 2.0f;
    }
}

// Transpose example showing proper indexing
__global__ void matrixTranspose(float *input, float *output, 
                               int width, int height) {
    // Calculate global thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Row-major to column-major transpose
        output[x * height + y] = input[y * width + x];
    }
}
```

## Summary
This chapter covered essential aspects of thread programming in CUDA:
- Thread hierarchy (warps, blocks, grids)
- Synchronization mechanisms
- Warp-level primitives and shuffles
- Occupancy considerations
- Handling divergent branching
- Advanced threading patterns
- Atomic operations
- Best practices for thread efficiency

Understanding these concepts is crucial for writing efficient CUDA kernels. The next chapter will explore kernel execution in detail.