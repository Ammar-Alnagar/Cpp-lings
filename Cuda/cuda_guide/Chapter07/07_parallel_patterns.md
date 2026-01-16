# Chapter 7: Parallel Programming Patterns in CUDA

## Overview
Parallel programming patterns are reusable solutions to common computational problems in parallel computing. Understanding these patterns is crucial for developing efficient CUDA applications. This chapter covers the most important parallel patterns and their CUDA implementations.

## Map Pattern

### Basic Map Operation
The map pattern applies a function to each element of a dataset independently.

```cuda
__global__ void mapKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Apply transformation function to each element
        output[idx] = input[idx] * input[idx] + 2.0f * input[idx] + 1.0f;
    }
}

// Generic map with function pointer (requires compute capability 3.2+)
__device__ float transformFunction(float x) {
    return x * x + sinf(x) + cosf(x);
}

__global__ void genericMapKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = transformFunction(input[idx]);
    }
}
```

### Vectorized Map for Better Memory Throughput
```cuda
// Process multiple elements per thread to increase arithmetic intensity
__global__ void vectorizedMapKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * 2.0f + 1.0f;
        
        // Process next element if available
        if (i + stride < n) {
            output[i + stride] = input[i + stride] * 2.0f + 1.0f;
        }
    }
}
```

## Reduce Pattern

### Basic Reduction
The reduce pattern combines elements of a dataset using an associative operator.

```cuda
__global__ void reduceBasic(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes one element
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
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
        output[blockIdx.x] = sdata[0];
    }
}
```

### Optimized Reduction
```cuda
__global__ void reduceOptimized(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Perform first level of reduction
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    sdata[tid] += (i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0.0f;
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
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

## Scan Pattern (Prefix Sum)

### Exclusive Scan
```cuda
__global__ void scanExclusive(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    temp[2 * thid] = (2 * thid < n) ? input[2 * thid] : 0.0f;
    temp[2 * thid + 1] = (2 * thid + 1 < n) ? input[2 * thid + 1] : 0.0f;
    
    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (thid == 0) {
        temp[n - 1] = 0;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results to device memory
    if (2 * thid < n) {
        output[2 * thid] = temp[2 * thid];
    }
    if (2 * thid + 1 < n) {
        output[2 * thid + 1] = temp[2 * thid + 1];
    }
}
```

## Histogram Pattern

### Simple Histogram
```cuda
__global__ void histogramKernel(unsigned char *input, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unsigned char value = input[idx];
        atomicAdd(&histogram[value], 1);
    }
}

// Optimized histogram using shared memory
__global__ void histogramOptimized(unsigned char *input, int *histogram, int n) {
    __shared__ int tempHistogram[256];
    
    // Initialize shared memory histogram
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x) {
        tempHistogram[i] = 0;
    }
    __syncthreads();
    
    // Process elements and update shared histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&tempHistogram[input[idx]], 1);
    }
    __syncthreads();
    
    // Merge shared histograms to global histogram
    for (int i = tid; i < 256; i += blockDim.x) {
        if (tempHistogram[i] > 0) {
            atomicAdd(&histogram[i], tempHistogram[i]);
        }
    }
}
```

## Matrix Multiplication Pattern

### Basic Matrix Multiplication
```cuda
__global__ void matMulBasic(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Tiled Matrix Multiplication
```cuda
#define TILE_SIZE 16

__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && (t * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < N && col < N) {
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
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Convolution Pattern

### Image Convolution
```cuda
#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH / 2

__global__ void convolutionKernel(float *input, float *output, float *mask,
                                 int width, int height) {
    __shared__ float tile[TILE_SIZE + 2 * MASK_RADIUS][TILE_SIZE + 2 * MASK_RADIUS];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Load data into shared memory tile with halo
    tile[ty][tx] = input[row * width + col];
    
    // Load border elements
    if (ty < MASK_RADIUS) {
        tile[ty + TILE_SIZE][tx] = (row + TILE_SIZE < height) ? 
                                   input[(row + TILE_SIZE) * width + col] : 0.0f;
    }
    if (ty >= TILE_SIZE - MASK_RADIUS) {
        tile[ty - TILE_SIZE][tx] = (row - TILE_SIZE >= 0) ? 
                                   input[(row - TILE_SIZE) * width + col] : 0.0f;
    }
    if (tx < MASK_RADIUS) {
        tile[ty][tx + TILE_SIZE] = (col + TILE_SIZE < width) ? 
                                   input[row * width + col + TILE_SIZE] : 0.0f;
    }
    if (tx >= TILE_SIZE - MASK_RADIUS) {
        tile[ty][tx - TILE_SIZE] = (col - TILE_SIZE >= 0) ? 
                                   input[row * width + col - TILE_SIZE] : 0.0f;
    }
    
    __syncthreads();
    
    // Perform convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                sum += tile[ty + i][tx + j] * mask[i * MASK_WIDTH + j];
            }
        }
        output[row * width + col] = sum;
    }
}
```

## Sort Pattern

### Bitonic Sort
```cuda
__device__ void compareAndSwap(float *data, int i, int j, int dir) {
    if ((data[i] > data[j]) == dir) {
        float temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

__global__ void bitonicSortStep(float *data, int j, int k, int dir, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int ixj = idx ^ j;
        if (ixj > idx && ixj < n) {
            compareAndSwap(data, idx, ixj, dir);
        }
    }
}

// Complete bitonic sort kernel
__global__ void bitonicSort(float *data, int k, int dir, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sz = 1 << k;
    
    if (idx < n) {
        for (int j = sz / 2; j > 0; j >>= 1) {
            int ixj = idx ^ j;
            if (ixj > idx && idx < n && ixj < n) {
                compareAndSwap(data, idx, ixj, dir);
            }
        }
    }
}
```

## Reduction-Map Pattern

### Combined Reduction and Mapping
```cuda
__global__ void reduceMapKernel(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Map and load into shared memory
    sdata[tid] = (idx < n) ? input[idx] * input[idx] : 0.0f;  // Map operation
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

## Stencil Pattern

### 2D 5-point Stencil
```cuda
__global__ void stencil5pt(float *input, float *output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        float center = input[row * width + col];
        float north = input[(row - 1) * width + col];
        float south = input[(row + 1) * width + col];
        float east = input[row * width + col + 1];
        float west = input[row * width + col - 1];
        
        // 5-point stencil computation
        output[row * width + col] = 0.2f * (center + north + south + east + west);
    }
}
```

## Gather-Scatter Pattern

### Indirect Memory Access
```cuda
__global__ void gatherScatter(float *input, float *output, int *indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Gather: read from scattered locations
        int srcIndex = indices[idx];
        float value = input[srcIndex];
        
        // Scatter: write to scattered locations
        int dstIndex = indices[idx] + 100;  // Example transformation
        output[dstIndex] = value * 2.0f;
    }
}
```

## Parallel Patterns Best Practices

### Memory Access Optimization
```cuda
// Coalesced access pattern
__global__ void coalescedAccess(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Consecutive threads access consecutive memory locations
        output[idx] = input[idx] * 2.0f;
    }
}

// Transpose example with proper indexing
__global__ void matrixTranspose(float *input, float *output, 
                               int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile to shared memory with coalesced access
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (y + j < height && x < width) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write transposed tile with coalesced access
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (y + j < width && x < height) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

### Occupancy Optimization
```cuda
// Use launch bounds to optimize for occupancy
__global__ void __launch_bounds__(256, 4)
optimizedPattern(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Computation that fits within register and shared memory limits
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
```

## Pattern Composition

### Combining Multiple Patterns
```cuda
__global__ void compositePattern(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 1: Map - Transform input
    float value = (idx < n) ? sqrtf(fabsf(input[idx])) : 0.0f;
    sdata[tid] = value;
    __syncthreads();
    
    // Step 2: Reduce - Compute local sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Step 3: Broadcast result and finalize
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];  // Store block result
    }
}
```

## Performance Considerations

### Arithmetic Intensity vs Memory Bandwidth
```cuda
// Low arithmetic intensity (memory-bound)
__global__ void memoryBound(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // Just one multiply per load/store
    }
}

// High arithmetic intensity (compute-bound)
__global__ void computeBound(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // Multiple operations per load/store
        for (int i = 0; i < 10; i++) {
            val = val * val + sinf(val) + cosf(val);
        }
        output[idx] = val;
    }
}
```

## Summary
This chapter covered essential parallel programming patterns in CUDA:
- Map pattern for element-wise transformations
- Reduce pattern for combining elements
- Scan pattern for prefix sums
- Histogram pattern for frequency counting
- Matrix multiplication patterns (basic and tiled)
- Convolution pattern for filtering operations
- Sorting patterns (bitonic sort)
- Stencil pattern for neighborhood operations
- Gather-scatter pattern for indirect access
- Best practices for optimization
- Pattern composition techniques
- Performance considerations

Understanding these patterns provides a foundation for solving a wide variety of parallel computing problems efficiently on GPU architectures. The next chapter will explore CUDA libraries and integration with other languages.