#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace hpc {

template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t size) : size_(size) {
        cudaMalloc(&ptr_, size_ * sizeof(T));
    }

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~DeviceMemory() {
        free();
    }

    void resize(size_t new_size) {
        if (new_size != size_) {
            free();
            if (new_size > 0) {
                cudaMalloc(&ptr_, new_size * sizeof(T));
                size_ = new_size;
            }
        }
    }

    void copy_from_host(const T* host_data, size_t count) {
        cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copy_to_host(T* host_data, size_t count) const {
        cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }

private:
    void free() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    T* ptr_;
    size_t size_;
};

template<typename T>
class UnifiedMemory {
public:
    UnifiedMemory() : ptr_(nullptr), size_(0) {}

    explicit UnifiedMemory(size_t size) : size_(size) {
        cudaMallocManaged(&ptr_, size_ * sizeof(T));
    }

    UnifiedMemory(const UnifiedMemory&) = delete;
    UnifiedMemory& operator=(const UnifiedMemory&) = delete;

    UnifiedMemory(UnifiedMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    UnifiedMemory& operator=(UnifiedMemory&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~UnifiedMemory() {
        free();
    }

    void resize(size_t new_size) {
        if (new_size != size_) {
            free();
            if (new_size > 0) {
                cudaMallocManaged(&ptr_, new_size * sizeof(T));
                size_ = new_size;
            }
        }
    }

    void prefetch_to_gpu(cudaStream_t stream = 0) {
        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(ptr_, size_ * sizeof(T), device, stream);
    }

    void prefetch_to_cpu(cudaStream_t stream = 0) {
        cudaMemPrefetchAsync(ptr_, size_ * sizeof(T), cudaCpuDeviceId, stream);
    }

    void advise_read_mostly() {
        cudaMemAdvise(ptr_, size_ * sizeof(T), cudaMemAdviseSetReadMostly, 0);
    }

    void advise_preferred_location_gpu() {
        int device;
        cudaGetDevice(&device);
        cudaMemAdvise(ptr_, size_ * sizeof(T), cudaMemAdviseSetPreferredLocation, device);
    }

    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("UnifiedMemory index out of bounds");
        }
        return ptr_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("UnifiedMemory index out of bounds");
        }
        return ptr_[index];
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }

private:
    void free() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    T* ptr_;
    size_t size_;
};

class CUDAStream {
public:
    CUDAStream() {
        cudaStreamCreate(&stream_);
    }

    ~CUDAStream() {
        cudaStreamDestroy(stream_);
    }

    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    CUDAStream(CUDAStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CUDAStream& operator=(CUDAStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    void synchronize() const {
        cudaStreamSynchronize(stream_);
    }

    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_;
};

class CUDAEvent {
public:
    explicit CUDAEvent(unsigned int flags = cudaEventDefault) {
        cudaEventCreateWithFlags(&event_, flags);
    }

    ~CUDAEvent() {
        cudaEventDestroy(event_);
    }

    CUDAEvent(const CUDAEvent&) = delete;
    CUDAEvent& operator=(const CUDAEvent&) = delete;

    void record(cudaStream_t stream = 0) const {
        cudaEventRecord(event_, stream);
    }

    void synchronize() const {
        cudaEventSynchronize(event_);
    }

    float elapsed_time(const CUDAEvent& start) const {
        float ms;
        cudaEventElapsedTime(&ms, start.event_, event_);
        return ms;
    }

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_;
};

} // namespace hpc
