#pragma once
#include "matrix.hpp"
#include "vector.hpp"
#include "utils.hpp"
#include "exceptions.hpp"

#ifdef __CUDACC__
#include "gpu_wrappers.hpp"
#include "gpu_kernels.hpp"
#include "optimized_kernels.hpp"
#include "stream_manager.hpp"
#endif

namespace hpc {

#ifdef __CUDACC__
template<typename T>
class MatrixGPU {
public:
    MatrixGPU(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), size_(rows * cols) {
        data_.resize(size_);
    }

    void upload(const Matrix<T>& cpu_matrix) {
        if (cpu_matrix.rows() != rows_ || cpu_matrix.cols() != cols_) {
            throw DimensionMismatchError(rows_, cols_, 
                                       cpu_matrix.rows(), cpu_matrix.cols());
        }
        std::copy(cpu_matrix.data(), cpu_matrix.data() + size_, data_.data());
        data_.prefetch_to_gpu();
    }

    Matrix<T> download() const {
        Matrix<T> result(rows_, cols_);
        cudaMemcpy(result.data(), data_.data(), 
                   size_ * sizeof(T), cudaMemcpyDeviceToHost);
        return result;
    }

    MatrixGPU multiply(const MatrixGPU& other) const {
        if (cols_ != other.rows_) {
            throw DimensionMismatchError(rows_, cols_, other.rows_, other.cols());
        }

        MatrixGPU<T> result(rows_, other.cols_);

        dim3 block(32, 32);
        dim3 grid((other.cols_ + block.x - 1) / block.x, 
                  (rows_ + block.y - 1) / block.y);

        gpu::matrix_multiply_simple_kernel<<<grid, block>>>(
            data_.data(), other.data_.data(), result.data_.data(),
            rows_, other.cols_, cols_);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CUDAError(err, "Matrix multiplication kernel failed");
        }

        return result;
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    size_t rows_;
    size_t cols_;
    size_t size_;
    UnifiedMemory<T> data_;
};
#endif

} // namespace hpc
