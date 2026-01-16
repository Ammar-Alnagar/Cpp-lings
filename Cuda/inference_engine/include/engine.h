// include/engine.h

#ifndef ENGINE_H
#define ENGINE_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Tensor structure
struct Tensor {
    float* data;          // GPU memory pointer
    int dims[4];          // Dimensions [batch, channel, height, width] or [batch, features]
    int ndim;             // Number of dimensions used
    size_t size;          // Total number of elements
    size_t bytes;         // Total bytes

    Tensor() : data(nullptr), ndim(0), size(0), bytes(0) {
        dims[0] = dims[1] = dims[2] = dims[3] = 0;
    }

    ~Tensor() {
        if (data) {
            cudaFree(data);
        }
    }

    void allocate(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
        dims[0] = d0; dims[1] = d1; dims[2] = d2; dims[3] = d3;
        ndim = (d3 > 1) ? 4 : (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
        size = d0 * d1 * d2 * d3;
        bytes = size * sizeof(float);

        CUDA_CHECK(cudaMalloc(&data, bytes));
    }

    void copyFromHost(const float* hostData) {
        CUDA_CHECK(cudaMemcpy(data, hostData, bytes, cudaMemcpyHostToDevice));
    }

    void copyToHost(float* hostData) {
        CUDA_CHECK(cudaMemcpy(hostData, data, bytes, cudaMemcpyDeviceToHost));
    }
};

// Layer types
enum LayerType {
    FC_LAYER,      // Fully connected
    CONV_LAYER,    // Convolution
    POOL_LAYER,    // Pooling
    ACTIVATION_LAYER  // Activation
};

// Base layer class
struct Layer {
    LayerType type;
    std::string name;

    Layer(LayerType t, const std::string& n) : type(t), name(n) {}
    virtual ~Layer() = default;

    virtual void forward(Tensor* input, Tensor* output, cublasHandle_t handle) = 0;
    virtual void loadWeights(const float* weights, size_t weightCount) = 0;
};

// Main engine class
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Load model from file
    bool loadModel(const std::string& modelPath);

    // Set input data
    void setInput(const float* inputData, int batchSize, int inputSize);

    // Run inference
    std::vector<float> runInference();

    // Get output
    const std::vector<float>& getOutput() const { return output; }

private:
    std::vector<Layer*> layers;
    cublasHandle_t cublasHandle;
    Tensor* inputTensor;
    Tensor* outputTensor;
    std::vector<float> output;

    void initializeCublas();
    void cleanup();
};

#endif // ENGINE_H