#include "gpu_kernels.hpp"
#include "gpu_wrappers.hpp"
#include "vector.hpp"
#include <cuda_runtime.h>

namespace hpc {

template<typename T>
void Vector<T>::add_gpu(const Vector& other) {
    if (size_ != other.size_) {
        throw std::runtime_error("Vector size mismatch");
    }

    Vector result(size_);

    UnifiedMemory<T> d_a(size_);
    UnifiedMemory<T> d_b(size_);
    UnifiedMemory<T> d_c(size_);

    std::copy(data_.get(), data_.get() + size_, d_a.data());
    std::copy(other.data_.get(), other.data_.get() + size_, d_b.data());

    d_a.prefetch_to_gpu();
    d_b.prefetch_to_gpu();

    constexpr int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;

    gpu::vector_add_kernel<<<grid_size, block_size>>>(d_a.data(), d_b.data(), 
                                                      d_c.data(), size_);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::copy(d_c.data(), d_c.data() + size_, result.data_.get());
    *this = result;
}

template<typename T>
void Vector<T>::scale_gpu(T scalar) {
    UnifiedMemory<T> d_a(size_);
    std::copy(data_.get(), data_.get() + size_, d_a.data());
    d_a.prefetch_to_gpu();

    constexpr int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;

    gpu::vector_scale_kernel<<<grid_size, block_size>>>(d_a.data(), scalar, 
                                                        d_a.data(), size_);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::copy(d_a.data(), d_a.data() + size_, data_.get());
}

template class Vector<float>;
template class Vector<double>;

} // namespace hpc
