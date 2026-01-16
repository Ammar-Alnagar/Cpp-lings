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

// Convolution kernel
__global__ void convolution(float *input, float *filter, float *output,
                           int input_h, int input_w, int input_c,
                           int filter_h, int filter_w, int output_c) {
    // Calculate output position
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;
    
    if (out_x >= (input_w - filter_w + 1) || 
        out_y >= (input_h - filter_h + 1) || 
        out_ch >= output_c) {
        return;
    }
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int f_y = 0; f_y < filter_h; f_y++) {
        for (int f_x = 0; f_x < filter_w; f_x++) {
            for (int ch = 0; ch < input_c; ch++) {
                int input_idx = ((ch * input_h + (out_y + f_y)) * input_w + (out_x + f_x));
                int filter_idx = ((((out_ch * input_c + ch) * filter_h + f_y) * filter_w + f_x));
                sum += input[input_idx] * filter[filter_idx];
            }
        }
    }
    
    int output_idx = ((out_ch * (input_h - filter_h + 1) + out_y) * (input_w - filter_w + 1) + out_x);
    output[output_idx] = sum;
}

// Optimized convolution with shared memory
#define TILE_SIZE 16
#define FILTER_SIZE 3

__global__ void convolution_optimized(float *input, float *filter, float *output,
                                     int input_h, int input_w, int input_c,
                                     int filter_h, int filter_w, int output_c) {
    __shared__ float tile[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1];
    __shared__ float filt[FILTER_SIZE][FILTER_SIZE];
    
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_ch = blockIdx.z;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load filter to shared memory
    if (threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE) {
        // Assuming filter is stored as [output_ch][input_ch][height][width]
        // For simplicity, we'll load the filter for the first input channel
        filt[threadIdx.y][threadIdx.x] = filter[out_ch * filter_h * filter_w + threadIdx.y * filter_w + threadIdx.x];
    }
    __syncthreads();
    
    // Load input tile with halo to shared memory
    for (int dy = 0; dy < TILE_SIZE + FILTER_SIZE - 1; dy += TILE_SIZE) {
        for (int dx = 0; dx < TILE_SIZE + FILTER_SIZE - 1; dx += TILE_SIZE) {
            int src_y = out_y + dy;
            int src_x = out_x + dx;
            
            if (src_y < input_h && src_x < input_w) {
                // For simplicity, using first channel
                tile[ty + dy][tx + dx] = input[src_y * input_w + src_x];
            } else {
                tile[ty + dy][tx + dx] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    // Perform convolution using shared memory
    if (out_x < (input_w - filter_w + 1) && out_y < (input_h - filter_h + 1)) {
        float sum = 0.0f;
        for (int f_y = 0; f_y < FILTER_SIZE; f_y++) {
            for (int f_x = 0; f_x < FILTER_SIZE; f_x++) {
                sum += tile[ty + f_y][tx + f_x] * filt[f_y][f_x];
            }
        }
        int output_idx = ((out_ch * (input_h - filter_h + 1) + out_y) * (input_w - filter_w + 1) + out_x);
        output[output_idx] = sum;
    }
}

// ReLU activation function
__global__ void relu_activation(float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Max pooling operation
__global__ void max_pooling(float *input, float *output,
                          int input_h, int input_w, int channels,
                          int pool_size) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (out_x >= input_w / pool_size || out_y >= input_h / pool_size || ch >= channels) {
        return;
    }
    
    float max_val = -INFINITY;
    
    // Find maximum in pooling window
    for (int py = 0; py < pool_size; py++) {
        for (int px = 0; px < pool_size; px++) {
            int src_y = out_y * pool_size + py;
            int src_x = out_x * pool_size + px;
            int src_idx = (ch * input_h + src_y) * input_w + src_x;
            
            if (input[src_idx] > max_val) {
                max_val = input[src_idx];
            }
        }
    }
    
    int out_h = input_h / pool_size;
    int out_w = input_w / pool_size;
    int output_idx = (ch * out_h + out_y) * out_w + out_x;
    output[output_idx] = max_val;
}

// Softmax activation function
__global__ void softmax_activation(float *input, float *output, int n) {
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

int main() {
    // Define network parameters
    const int INPUT_HEIGHT = 32;
    const int INPUT_WIDTH = 32;
    const int INPUT_CHANNELS = 3;
    const int FILTER_HEIGHT = 3;
    const int FILTER_WIDTH = 3;
    const int OUTPUT_CHANNELS = 16;
    
    // Calculate sizes
    const int input_size = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
    const int filter_size = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS * OUTPUT_CHANNELS;
    const int output_height = INPUT_HEIGHT - FILTER_HEIGHT + 1;
    const int output_width = INPUT_WIDTH - FILTER_WIDTH + 1;
    const int output_size = output_height * output_width * OUTPUT_CHANNELS;
    
    // Allocate host memory
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_filter = (float*)malloc(filter_size * sizeof(float));
    float *h_output = (float*)malloc(output_size * sizeof(float));
    
    // Initialize host data
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (float)(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < filter_size; i++) {
        h_filter[i] = (float)(rand() % 10) / 100.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    gpuErrchk(cudaMalloc(&d_input, input_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_filter, filter_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_output, output_size * sizeof(float)));
    
    // Copy data from host to device
    gpuErrchk(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filter, h_filter, filter_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x,
                  (output_height + blockSize.y - 1) / blockSize.y,
                  OUTPUT_CHANNELS);
    
    // Launch convolution kernel
    convolution<<<gridSize, blockSize>>>(d_input, d_filter, d_output,
                                         INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
                                         FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_CHANNELS);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Apply activation function (ReLU)
    int total_elements = output_size;
    int linear_block_size = 256;
    int linear_grid_size = (total_elements + linear_block_size - 1) / linear_block_size;
    
    relu_activation<<<linear_grid_size, linear_block_size>>>(d_output, total_elements);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Perform pooling operation
    const int pool_size = 2;
    const int pooled_height = output_height / pool_size;
    const int pooled_width = output_width / pool_size;
    const int pooled_size = pooled_height * pooled_width * OUTPUT_CHANNELS;
    
    dim3 pool_gridSize((pooled_width + blockSize.x - 1) / blockSize.x,
                       (pooled_height + blockSize.y - 1) / blockSize.y,
                       OUTPUT_CHANNELS);
    
    // Note: For pooling, we need to allocate a new output buffer or handle in-place differently
    float *d_pooled;
    gpuErrchk(cudaMalloc(&d_pooled, pooled_size * sizeof(float)));
    
    max_pooling<<<pool_gridSize, blockSize>>>(d_output, d_pooled,
                                             output_height, output_width, OUTPUT_CHANNELS,
                                             pool_size);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host
    gpuErrchk(cudaMemcpy(h_output, d_pooled, pooled_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print some results for verification
    printf("CNN operations completed.\n");
    printf("Input size: %dx%dx%d\n", INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS);
    printf("Output size after conv+relu+pool: %dx%dx%d\n", 
           pooled_height, pooled_width, OUTPUT_CHANNELS);
    
    printf("First 5x5 elements of first channel after pooling:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int idx = i * pooled_width + j;  // First channel
            printf("%.3f ", h_output[idx]);
        }
        printf("\n");
    }
    
    // Cleanup allocated memory
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_output));
    gpuErrchk(cudaFree(d_pooled));
    
    free(h_input);
    free(h_filter);
    free(h_output);
    
    return 0;
}