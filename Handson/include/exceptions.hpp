#pragma once
#include <stdexcept>
#include <string>

namespace hpc {

class HPCException : public std::runtime_error {
public:
    explicit HPCException(const std::string& msg) 
        : std::runtime_error(msg) {}
};

class MatrixError : public HPCException {
public:
    explicit MatrixError(const std::string& msg) : HPCException(msg) {}
};

class DimensionMismatchError : public MatrixError {
public:
    DimensionMismatchError(size_t rows1, size_t cols1, size_t rows2, size_t cols2)
        : MatrixError("Dimension mismatch: " + std::to_string(rows1) + "x" + 
                      std::to_string(cols1) + " vs " + std::to_string(rows2) + 
                      "x" + std::to_string(cols2)) {}
};

class MemoryAllocationError : public HPCException {
public:
    explicit MemoryAllocationError(size_t size)
        : HPCException("Failed to allocate " + std::to_string(size) + " bytes") {}
};

} // namespace hpc

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace hpc {

class CUDAError : public HPCException {
public:
    explicit CUDAError(cudaError_t error, const std::string& msg)
        : HPCException(msg + " (CUDA Error: " + std::string(cudaGetErrorString(error)) + ")"),
          error_code_(error) {}

    cudaError_t error_code() const noexcept { return error_code_; }

private:
    cudaError_t error_code_;
};

inline void check_cuda(cudaError_t error, const std::string& msg = "CUDA operation failed") {
    if (error != cudaSuccess) {
        throw CUDAError(error, msg);
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        hpc::check_cuda(err, #call); \
    } while(0)

} // namespace hpc
#endif
