// src/layers.cu

#include "../include/layers.h"
#include <cstring>
#include <algorithm>
#include <climits>

// FCLayer implementation
FCLayer::FCLayer(int inSize, int outSize) 
    : Layer(FC_LAYER, "FC"), 
      inputSize(inSize), 
      outputSize(outSize) {
    weights.allocate(inputSize, outputSize);
    biases.allocate(1, outputSize);
}

void FCLayer::forward(Tensor* input, Tensor* output, cublasHandle_t handle) {
    // Ensure output tensor is properly sized
    if (output->dims[0] != input->dims[0] || output->dims[1] != outputSize) {
        delete output;
        output = new Tensor();
        output->allocate(input->dims[0], outputSize);
    }
    
    // Matrix multiplication: output = input * weights + biases
    const float alpha = 1.0f, beta = 1.0f;
    int batchSize = input->dims[0];
    
    for (int b = 0; b < batchSize; b++) {
        cublasSgemv(handle, CUBLAS_OP_N,
                    outputSize, inputSize,
                    &alpha,
                    weights.data + b * inputSize * outputSize, outputSize,
                    input->data + b * inputSize, 1,
                    &beta,
                    output->data + b * outputSize, 1);
    }
}

void FCLayer::loadWeights(const float* weightsData, size_t weightCount) {
    size_t weightSize = inputSize * outputSize;
    size_t biasSize = outputSize;
    
    if (weightCount != weightSize + biasSize) {
        fprintf(stderr, "Weight count mismatch in FC layer\n");
        return;
    }
    
    // Copy weights
    CUDA_CHECK(cudaMemcpy(weights.data, weightsData, 
                         weightSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy biases
    CUDA_CHECK(cudaMemcpy(biases.data, weightsData + weightSize, 
                         biasSize * sizeof(float), cudaMemcpyHostToDevice));
}

// ConvLayer implementation
ConvLayer::ConvLayer(int inCh, int outCh, int kH, int kW, int sH, int sW, int pH, int pW)
    : Layer(CONV_LAYER, "Conv"),
      inputChannels(inCh), outputChannels(outCh),
      kernelH(kH), kernelW(kW),
      strideH(sH), strideW(sW),
      padH(pH), padW(pW) {}

void ConvLayer::forward(Tensor* input, Tensor* output, cublasHandle_t handle) {
    // Calculate output dimensions
    outputH = (inputH + 2 * padH - kernelH) / strideH + 1;
    outputW = (inputW + 2 * padW - kernelW) / strideW + 1;
    
    // Allocate output tensor if needed
    if (output->dims[0] != input->dims[0] || output->dims[1] != outputChannels ||
        output->dims[2] != outputH || output->dims[3] != outputW) {
        delete output;
        output = new Tensor();
        output->allocate(input->dims[0], outputChannels, outputH, outputW);
    }
    
    // This is a simplified implementation
    // In a real implementation, you'd use cuDNN or implement im2col properly
    // For now, we'll use a basic approach
    
    // Temporary tensor for reshaped input (im2col format)
    int patchCount = outputH * outputW;
    int patchSize = inputChannels * kernelH * kernelW;
    Tensor im2colTemp;
    im2colTemp.allocate(input->dims[0], patchCount, patchSize);
    
    // Perform im2col transformation
    im2col(input->data, im2colTemp.data, input->dims[0]);
    
    // Reshape weights for matrix multiplication
    Tensor reshapedWeights;
    reshapedWeights.allocate(patchSize, outputChannels);
    
    // Copy and reshape weights
    // This is a simplified approach - in reality, weights need to be reordered
    CUDA_CHECK(cudaMemcpy(reshapedWeights.data, weights.data, 
                         weights.size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Perform matrix multiplication: output_patches = im2col_patches * weights
    const float alpha = 1.0f, beta = 0.0f;
    int batchSize = input->dims[0];
    
    for (int b = 0; b < batchSize; b++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    outputChannels, patchCount, patchSize,
                    &alpha,
                    reshapedWeights.data, outputChannels,
                    im2colTemp.data + b * patchCount * patchSize, patchSize,
                    &beta,
                    output->data + b * outputChannels * patchCount, outputChannels);
    }
    
    // Add biases
    // This would require a kernel to broadcast biases across spatial dimensions
}

void ConvLayer::im2col(const float* input, float* output, int batchSize) {
    // Simplified im2col implementation
    // This is a CPU implementation for illustration - a real implementation
    // would use optimized GPU kernels
    for (int b = 0; b < batchSize; b++) {
        for (int oh = 0; oh < outputH; oh++) {
            for (int ow = 0; ow < outputW; ow++) {
                int patchIdx = oh * outputW + ow;
                for (int ic = 0; ic < inputChannels; ic++) {
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            int ih = oh * strideH + kh - padH;
                            int iw = ow * strideW + kw - padW;
                            
                            float val = 0.0f;
                            if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW) {
                                int inputIdx = (((b * inputChannels + ic) * inputH + ih) * inputW + iw);
                                val = input[inputIdx];
                            }
                            
                            int outputIdx = (b * outputH * outputW + patchIdx) * (inputChannels * kernelH * kernelW) +
                                           (ic * kernelH + kh) * kernelW + kw;
                            output[outputIdx] = val;
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::loadWeights(const float* weightsData, size_t weightCount) {
    size_t weightSize = inputChannels * outputChannels * kernelH * kernelW;
    size_t biasSize = outputChannels;
    
    if (weightCount != weightSize + biasSize) {
        fprintf(stderr, "Weight count mismatch in Conv layer\n");
        return;
    }
    
    // Copy weights (need to reorder from [outCh][inCh][H][W] to internal format)
    CUDA_CHECK(cudaMemcpy(weights.data, weightsData, 
                         weightSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy biases
    CUDA_CHECK(cudaMemcpy(biases.data, weightsData + weightSize, 
                         biasSize * sizeof(float), cudaMemcpyHostToDevice));
}

// PoolLayer implementation
PoolLayer::PoolLayer(int kH, int kW, int sH, int sW)
    : Layer(POOL_LAYER, "Pool"),
      kernelH(kH), kernelW(kW),
      strideH(sH), strideW(sW) {}

void PoolLayer::forward(Tensor* input, Tensor* output, cublasHandle_t handle) {
    // Calculate output dimensions
    outputH = (input->dims[2] - kernelH) / strideH + 1;
    outputW = (input->dims[3] - kernelW) / strideW + 1;
    
    // Allocate output tensor if needed
    if (output->dims[0] != input->dims[0] || output->dims[1] != input->dims[1] ||
        output->dims[2] != outputH || output->dims[3] != outputW) {
        delete output;
        output = new Tensor();
        output->allocate(input->dims[0], input->dims[1], outputH, outputW);
    }
    
    // Launch max pooling kernel
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((outputW + blockSize.x - 1) / blockSize.x,
                  (outputH + blockSize.y - 1) / blockSize.y,
                  input->dims[1]);
    
    maxPoolKernel<<<gridSize, blockSize>>>(
        input->data, output->data,
        input->dims[2], input->dims[3], input->dims[1],
        outputH, outputW,
        kernelH, kernelW,
        strideH, strideW
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void PoolLayer::loadWeights(const float* weights, size_t weightCount) {
    // Pooling layers don't have trainable weights
}

// CUDA kernel for max pooling
__global__ void maxPoolKernel(
    const float* input, float* output,
    int inputH, int inputW, int channels,
    int outputH, int outputW,
    int kernelH, int kernelW,
    int strideH, int strideW
) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (outX >= outputW || outY >= outputH || ch >= channels) {
        return;
    }
    
    float maxVal = -FLT_MAX;
    
    // Find maximum in pooling window
    for (int ky = 0; ky < kernelH; ky++) {
        for (int kx = 0; kx < kernelW; kx++) {
            int inY = outY * strideH + ky;
            int inX = outX * strideW + kx;
            
            if (inY < inputH && inX < inputW) {
                int inputIdx = ((ch * inputH + inY) * inputW + inX);
                float val = input[inputIdx];
                if (val > maxVal) {
                    maxVal = val;
                }
            }
        }
    }
    
    int outputIdx = ((ch * outputH + outY) * outputW + outX);
    output[outputIdx] = maxVal;
}