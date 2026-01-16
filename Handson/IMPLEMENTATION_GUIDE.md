# Comprehensive Implementation Guide

This guide provides step-by-step instructions for building the Modern C++ and CUDA Hands-On Project. Follow these sections in order to build the complete HPC Toolkit.

## Table of Contents

1. [Phase 1: Build System and Project Setup](#phase-1-build-system-and-project-setup)
2. [Phase 2: Core Utilities and Exception Handling](#phase-2-core-utilities-and-exception-handling)
3. [Phase 3: CPU Matrix Implementation](#phase-3-cpu-matrix-implementation)
4. [Phase 4: CPU Vector Implementation](#phase-4-cpu-vector-implementation)
5. [Phase 5: CUDA Memory Management](#phase-5-cuda-memory-management)
6. [Phase 6: Basic GPU Kernels](#phase-6-basic-gpu-kernels)
7. [Phase 7: Optimized GPU Matrix Multiplication](#phase-7-optimized-gpu-matrix-multiplication)
8. [Phase 8: Parallel Primitives](#phase-8-parallel-primitives)
9. [Phase 9: CUDA Streams and Events](#phase-9-cuda-streams-and-events)
10. [Phase 10: Advanced CUDA Features](#phase-10-advanced-cuda-features)
11. [Phase 11: Integration and High-Level API](#phase-11-integration-and-high-level-api)
12. [Phase 12: Example Programs](#phase-12-example-programs)

---

## Phase 1: Build System and Project Setup

### Step 1.1: Create CMakeLists.txt

Create the main CMakeLists.txt file that handles both C++ and CUDA compilation.

**File: `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.25)
project(HPCToolkit LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")  # V100, A100, H100

# Find required packages
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(CPU_SOURCES
    src/cpu_matrix.cpp
    src/cpu_vector.cpp
    src/utils.cpp
)

set(GPU_SOURCES
    src/gpu_matrix.cu
    src/gpu_vector.cu
    src/reductions.cu
    src/scan.cu
)

set(EXAMPLE_SOURCES
    examples/example1_basics.cpp
    examples/example2_matrix_ops.cpp
    examples/example3_gpu_acceleration.cpp
    examples/example4_streams.cpp
    examples/example5_advanced.cpp
)

# Create library
add_library(hpc_toolkit SHARED ${CPU_SOURCES} ${GPU_SOURCES})
target_link_libraries(hpc_toolkin ${CUDA_LIBRARIES})

# Create executables
foreach(example ${EXAMPLE_SOURCES})
    get_filename_component(exe_name ${example} NAME_WE)
    add_executable(${exe_name} ${example})
    target_link_libraries(${exe_name} hpc_toolkit)
endforeach()

# Benchmark executable
add_executable(benchmark_suite benchmarks/benchmark_suite.cpp)
target_link_libraries(benchmark_suite hpc_toolkit)

# Test executable
enable_testing()
add_executable(test_suite tests/test_suite.cpp)
target_link_libraries(test_suite hpc_toolkit)
add_test(NAME test_suite COMMAND test_suite)
```

### Step 1.2: Verify Build System

Test the build system:

```bash
mkdir build && cd build
cmake ..
make
```

---

## Phase 2: Core Utilities and Exception Handling

### Step 2.1: Create Exception Classes

**File: `include/exceptions.hpp`**

```cpp
#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace hpc {

class HPCException : public std::runtime_error {
public:
    explicit HPCException(const std::string& msg) 
        : std::runtime_error(msg) {}
};

class CUDAError : public HPCException {
public:
    explicit CUDAError(cudaError_t error, const std::string& msg)
        : HPCException(msg + " (CUDA Error: " + std::string(cudaGetErrorString(error)) + ")"),
          error_code_(error) {}

    cudaError_t error_code() const noexcept { return error_code_; }

private:
    cudaError_t error_code_;
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
```

### Step 2.2: Create Utility Functions

**File: `include/utils.hpp`**

```cpp
#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace hpc {

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) 
        : name_(name), timer_() {}

    ~ScopedTimer() {
        std::cout << name_ << ": " << timer_.elapsed_ms() << " ms" << std::endl;
    }

private:
    std::string name_;
    Timer timer_;
};

inline void print_separator(char c = '=', size_t width = 80) {
    std::cout << std::string(width, c) << std::endl;
}

inline void print_header(const std::string& title) {
    print_separator('=');
    std::cout << title << std::endl;
    print_separator('=');
}

template<typename T>
void print_matrix(const T* data, size_t rows, size_t cols, size_t max_rows = 10, size_t max_cols = 10) {
    size_t print_rows = std::min(rows, max_rows);
    size_t print_cols = std::min(cols, max_cols);

    std::cout << "Matrix " << rows << "x" << cols << ":" << std::endl;
    for (size_t i = 0; i < print_rows; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < print_cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << data[i * cols + j];
            if (j < print_cols - 1) std::cout << " ";
        }
        if (cols > max_cols) std::cout << " ...";
        std::cout << "]" << std::endl;
    }
    if (rows > max_rows) {
        std::cout << "..." << std::endl;
    }
}

inline std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    size_t unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

} // namespace hpc
```

---

## Phase 3: CPU Matrix Implementation

### Step 3.1: Matrix Header

**File: `include/matrix.hpp`**

```cpp
#pragma once
#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <span>
#include <concepts>
#include "exceptions.hpp"
#include "utils.hpp"

namespace hpc {

template<typename T>
class Matrix {
public:
    // Constructors
    Matrix() : rows_(0), cols_(0), data_(nullptr), size_(0), capacity_(0) {}

    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), size_(rows * cols) {
        allocate();
    }

    Matrix(size_t rows, size_t cols, T value)
        : Matrix(rows, cols) {
        std::fill(data_.get(), data_.get() + size_, value);
    }

    Matrix(const Matrix& other) 
        : rows_(other.rows_), cols_(other.cols_), size_(other.size_) {
        allocate();
        std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
    }

    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), 
          data_(std::move(other.data_)), 
          size_(other.size_), capacity_(other.capacity_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // Assignment operators
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Matrix temp(other);
            swap(*this, temp);
        }
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            swap(*this, other);
        }
        return *this;
    }

    // Accessors
    T& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            throw MatrixError("Index out of bounds");
        }
        return data_[row * cols_ + col];
    }

    const T& operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw MatrixError("Index out of bounds");
        }
        return data_[row * cols_ + col];
    }

    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return size_; }

    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols) {
        return Matrix(rows, cols, T{0});
    }

    static Matrix ones(size_t rows, size_t cols) {
        return Matrix(rows, cols, T{1});
    }

    static Matrix identity(size_t size) {
        Matrix result(size, size, T{0});
        for (size_t i = 0; i < size; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

    static Matrix random(size_t rows, size_t cols, T min = -1.0, T max = 1.0) {
        Matrix result(rows, cols);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);

        for (size_t i = 0; i < result.size_; ++i) {
            result.data_[i] = dist(gen);
        }
        return result;
    }

    // Matrix operations
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        check_dimensions(other);
        Matrix result(rows_, cols_);
        std::transform(data_.get(), data_.get() + size_, 
                       other.data_.get(), result.data_.get(), 
                       std::plus<T>());
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        check_dimensions(other);
        Matrix result(rows_, cols_);
        std::transform(data_.get(), data_.get() + size_, 
                       other.data_.get(), result.data_.get(), 
                       std::minus<T>());
        return result;
    }

    Matrix operator*(T scalar) const {
        Matrix result(rows_, cols_);
        std::transform(data_.get(), data_.get() + size_, 
                       result.data_.get(), 
                       [scalar](T val) { return val * scalar; });
        return result;
    }

    // Matrix multiplication (CPU)
    Matrix multiply_cpu(const Matrix& other) const;

    // Element-wise operations
    Matrix elementwise_multiply(const Matrix& other) const {
        check_dimensions(other);
        Matrix result(rows_, cols_);
        std::transform(data_.get(), data_.get() + size_, 
                       other.data_.get(), result.data_.get(), 
                       std::multiplies<T>());
        return result;
    }

    // Utility functions
    void fill(T value) {
        std::fill(data_.get(), data_.get() + size_, value);
    }

    void resize(size_t new_rows, size_t new_cols) {
        size_t new_size = new_rows * new_cols;
        if (new_size > capacity_) {
            reallocate(new_size);
        }
        rows_ = new_rows;
        cols_ = new_cols;
        size_ = new_size;
    }

    void print(const std::string& name = "Matrix") const {
        print_header(name);
        print_matrix(data_.get(), rows_, cols_);
    }

    // Friends
    friend void swap(Matrix& a, Matrix& b) noexcept {
        using std::swap;
        swap(a.rows_, b.rows_);
        swap(a.cols_, b.cols_);
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
        swap(a.capacity_, b.capacity_);
    }

    friend bool operator==(const Matrix& a, const Matrix& b) {
        if (a.rows_ != b.rows_ || a.cols_ != b.cols_) {
            return false;
        }
        return std::equal(a.data_.get(), a.data_.get() + a.size_, 
                          b.data_.get(), [](T x, T y) {
                              return std::abs(x - y) < 1e-6;
                          });
    }

private:
    void allocate() {
        if (size_ > 0) {
            data_ = std::make_unique<T[]>(size_);
            capacity_ = size_;
        }
    }

    void reallocate(size_t new_size) {
        auto new_data = std::make_unique<T[]>(new_size);
        std::copy(data_.get(), data_.get() + std::min(size_, new_size), new_data.get());
        data_ = std::move(new_data);
        capacity_ = new_size;
    }

    void check_dimensions(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw DimensionMismatchError(rows_, cols_, other.rows_, other.cols_);
        }
    }

    size_t rows_;
    size_t cols_;
    std::unique_ptr<T[]> data_;
    size_t size_;
    size_t capacity_;
};

} // namespace hpc
```

### Step 3.2: Matrix CPU Implementation

**File: `src/cpu_matrix.cpp`**

```cpp
#include "matrix.hpp"
#include <algorithm>
#include <stdexcept>

namespace hpc {

template<typename T>
Matrix<T> Matrix<T>::multiply_cpu(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw DimensionMismatchError(rows_, cols_, other.rows_, other.cols_);
    }

    Matrix result(rows_, other.cols_, T{0});

    // Standard O(n^3) matrix multiplication
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = 0; k < cols_; ++k) {
            T a_ik = (*this)(i, k);
            for (size_t j = 0; j < other.cols_; ++j) {
                result(i, j) += a_ik * other(k, j);
            }
        }
    }

    return result;
}

// Explicit template instantiations
template class Matrix<float>;
template class Matrix<double>;

} // namespace hpc
```

---

## Phase 4: CPU Vector Implementation

### Step 4.1: Vector Header

**File: `include/vector.hpp`**

```cpp
#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <concepts>
#include <span>
#include "exceptions.hpp"
#include "utils.hpp"

namespace hpc {

template<typename T>
class Vector {
public:
    using iterator = T*;
    using const_iterator = const T*;

    // Constructors
    Vector() : size_(0), capacity_(0), data_(nullptr) {}

    explicit Vector(size_t size) : size_(size) {
        allocate();
    }

    Vector(size_t size, T value) : Vector(size) {
        std::fill(data_.get(), data_.get() + size_, value);
    }

    Vector(const std::vector<T>& vec) : size_(vec.size()) {
        allocate();
        std::copy(vec.begin(), vec.end(), data_.get());
    }

    Vector(const Vector& other) : size_(other.size_) {
        allocate();
        std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
    }

    Vector(Vector&& other) noexcept
        : size_(other.size_), capacity_(other.capacity_),
          data_(std::move(other.data_)) {
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // Assignment
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            Vector temp(other);
            swap(*this, temp);
        }
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            swap(*this, other);
        }
        return *this;
    }

    // Accessors
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Vector index out of bounds");
        }
        return data_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Vector index out of bounds");
        }
        return data_[index];
    }

    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }

    iterator begin() noexcept { return data_.get(); }
    iterator end() noexcept { return data_.get() + size_; }
    const_iterator begin() const noexcept { return data_.get(); }
    const_iterator end() const noexcept { return data_.get() + size_; }
    const_iterator cbegin() const noexcept { return data_.get(); }
    const_iterator cend() const noexcept { return data_.get() + size_; }

    std::span<T> span() noexcept { return std::span<T>(data_.get(), size_); }
    std::span<const T> span() const noexcept { return std::span<const T>(data_.get(), size_); }

    // Factory methods
    static Vector zeros(size_t size) {
        return Vector(size, T{0});
    }

    static Vector ones(size_t size) {
        return Vector(size, T{1});
    }

    static Vector random(size_t size, T min = -1.0, T max = 1.0) {
        Vector result(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);

        std::generate(result.begin(), result.end(), [&dist, &gen]() {
            return dist(gen);
        });
        return result;
    }

    // Vector operations
    Vector operator+(const Vector& other) const {
        if (size_ != other.size_) {
            throw std::runtime_error("Vector size mismatch");
        }
        Vector result(size_);
        std::transform(begin(), end(), other.begin(), result.begin(), std::plus<T>());
        return result;
    }

    Vector operator-(const Vector& other) const {
        if (size_ != other.size_) {
            throw std::runtime_error("Vector size mismatch");
        }
        Vector result(size_);
        std::transform(begin(), end(), other.begin(), result.begin(), std::minus<T>());
        return result;
    }

    T dot(const Vector& other) const {
        if (size_ != other.size_) {
            throw std::runtime_error("Vector size mismatch");
        }
        return std::inner_product(begin(), end(), other.begin(), T{0});
    }

    T norm() const {
        return std::sqrt(dot(*this));
    }

    Vector normalized() const {
        T n = norm();
        if (n == T{0}) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        Vector result(size_);
        std::transform(begin(), end(), result.begin(), [n](T val) { return val / n; });
        return result;
    }

    // Reductions
    T sum() const {
        return std::accumulate(begin(), end(), T{0});
    }

    T mean() const {
        return size_ > 0 ? sum() / static_cast<T>(size_) : T{0};
    }

    T min() const {
        return *std::min_element(begin(), end());
    }

    T max() const {
        return *std::max_element(begin(), end());
    }

    // Sorting
    void sort() {
        std::sort(begin(), end());
    }

    void sort_descending() {
        std::sort(begin(), end(), std::greater<T>());
    }

    // Utility
    void fill(T value) {
        std::fill(begin(), end(), value);
    }

    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reallocate(new_size);
        }
        size_ = new_size;
    }

    void push_back(T value) {
        if (size_ >= capacity_) {
            size_t new_capacity = capacity_ == 0 ? 1 : capacity_ * 2;
            reallocate(new_capacity);
        }
        data_[size_++] = value;
    }

    void print(const std::string& name = "Vector") const {
        std::cout << name << " [" << size_ << "]: [";
        size_t print_size = std::min(size_, size_t(10));
        for (size_t i = 0; i < print_size; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data_[i];
            if (i < print_size - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    friend void swap(Vector& a, Vector& b) noexcept {
        using std::swap;
        swap(a.size_, b.size_);
        swap(a.capacity_, b.capacity_);
        swap(a.data_, b.data_);
    }

private:
    void allocate() {
        if (size_ > 0) {
            data_ = std::make_unique<T[]>(size_);
            capacity_ = size_;
        }
    }

    void reallocate(size_t new_capacity) {
        auto new_data = std::make_unique<T[]>(new_capacity);
        std::copy(data_.get(), data_.get() + size_, new_data.get());
        data_ = std::move(new_data);
        capacity_ = new_capacity;
    }

    size_t size_;
    size_t capacity_;
    std::unique_ptr<T[]> data_;
};

} // namespace hpc
```

---

## Phase 5: CUDA Memory Management

### Step 5.1: CUDA Memory Wrappers

**File: `include/gpu_wrappers.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "exceptions.hpp"

namespace hpc {

// RAII wrapper for CUDA device memory
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
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
                CUDA_CHECK(cudaMalloc(&ptr_, new_size * sizeof(T)));
                size_ = new_size;
            }
        }
    }

    void copy_from_host(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), 
                             cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), 
                             cudaMemcpyDeviceToHost));
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

// RAII wrapper for CUDA unified memory
template<typename T>
class UnifiedMemory {
public:
    UnifiedMemory() : ptr_(nullptr), size_(0) {}

    explicit UnifiedMemory(size_t size) : size_(size) {
        CUDA_CHECK(cudaMallocManaged(&ptr_, size_ * sizeof(T)));
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
                CUDA_CHECK(cudaMallocManaged(&ptr_, new_size * sizeof(T)));
                size_ = new_size;
            }
        }
    }

    void prefetch_to_gpu(cudaStream_t stream = 0) {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaMemPrefetchAsync(ptr_, size_ * sizeof(T), 
                                        device, stream));
    }

    void prefetch_to_cpu(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemPrefetchAsync(ptr_, size_ * sizeof(T), 
                                        cudaCpuDeviceId, stream));
    }

    void advise_read_mostly() {
        CUDA_CHECK(cudaMemAdvise(ptr_, size_ * sizeof(T), 
                                cudaMemAdviseSetReadMostly, 0));
    }

    void advise_preferred_location_gpu() {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaMemAdvise(ptr_, size_ * sizeof(T), 
                                cudaMemAdviseSetPreferredLocation, device));
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

// RAII wrapper for CUDA streams
class CUDAStream {
public:
    CUDAStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
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
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_;
};

// RAII wrapper for CUDA events
class CUDAEvent {
public:
    explicit CUDAEvent(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CUDAEvent() {
        cudaEventDestroy(event_);
    }

    CUDAEvent(const CUDAEvent&) = delete;
    CUDAEvent& operator=(const CUDAEvent&) = delete;

    void record(cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_time(const CUDAEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_;
};

} // namespace hpc
```

---

## Phase 6: Basic GPU Kernels

### Step 6.1: Basic Vector Kernels

**File: `include/gpu_kernels.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>

namespace hpc::gpu {

// Vector addition kernel
template<typename T>
__global__ void vector_add_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector subtraction kernel
template<typename T>
__global__ void vector_subtract_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Scalar multiplication kernel
template<typename T>
__global__ void vector_scale_kernel(const T* a, T scalar, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

// Matrix initialization kernel
template<typename T>
__global__ void matrix_init_kernel(T* data, size_t rows, size_t cols, T value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;
    if (idx < total) {
        data[idx] = value;
    }
}

// Matrix transpose kernel (coalesced)
template<typename T>
__global__ void matrix_transpose_kernel(const T* input, T* output, 
                                       size_t rows, size_t cols) {
    __shared__ T tile[32][32];

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Simple matrix multiplication kernel
template<typename T>
__global__ void matrix_multiply_simple_kernel(const T* A, const T* B, T* C,
                                               size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = T{0};
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

} // namespace hpc::gpu
```

### Step 6.2: GPU Vector Implementation

**File: `src/gpu_vector.cu`**

```cpp
#include "vector.hpp"
#include "gpu_wrappers.hpp"
#include "gpu_kernels.hpp"
#include <cuda_runtime.h>

namespace hpc {

template<typename T>
Vector<T> Vector<T>::multiply_gpu(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::runtime_error("Vector size mismatch");
    }

    Vector result(size_);

    // Allocate device memory
    DeviceMemory<T> d_a(size_);
    DeviceMemory<T> d_b(size_);
    DeviceMemory<T> d_c(size_);

    // Copy data to device
    d_a.copy_from_host(data_.get(), size_);
    d_b.copy_from_host(other.data_.get(), size_);

    // Launch kernel
    constexpr int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;

    gpu::vector_add_kernel<<<grid_size, block_size>>>(d_a.get(), d_b.get(), 
                                                      d_c.get(), size_);

    // Copy result back
    d_c.copy_to_host(result.data_.get(), size_);

    return result;
}

template<typename T>
void Vector<T>::scale_gpu(T scalar) {
    DeviceMemory<T> d_a(size_);
    d_a.copy_from_host(data_.get(), size_);

    constexpr int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;

    gpu::vector_scale_kernel<<<grid_size, block_size>>>(d_a.get(), scalar, 
                                                        d_a.get(), size_);

    d_a.copy_to_host(data_.get(), size_);
}

// Explicit template instantiations
template class Vector<float>;
template class Vector<double>;

} // namespace hpc
```

---

## Phase 7: Optimized GPU Matrix Multiplication

### Step 7.1: Tiled Matrix Multiplication Kernel

**File: `include/optimized_kernels.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>

namespace hpc::gpu {

// Tiled matrix multiplication with shared memory
template<typename T, int TILE_SIZE>
__global__ void matrix_multiply_tiled_kernel(const T* A, const T* B, T* C,
                                               size_t M, size_t N, size_t K) {
    __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = T{0};

    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile A
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = 
                A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = T{0};
        }

        // Load tile B
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = T{0};
        }

        __syncthreads();

        // Compute partial dot product
        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matrix multiplication with register tiling (2x2)
template<typename T>
__global__ void matrix_multiply_register_tiled_kernel(const T* A, const T* B, T* C,
                                                       size_t M, size_t N, size_t K) {
    constexpr int TILE_SIZE = 32;
    __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

    size_t global_row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    size_t global_col = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T sum00 = T{0}, sum01 = T{0}, sum10 = T{0}, sum11 = T{0};

    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles
        if (global_row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = 
                A[global_row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = T{0};
        }

        if (global_row + blockDim.y < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y + blockDim.y][threadIdx.x] = 
                A[(global_row + blockDim.y) * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y + blockDim.y][threadIdx.x] = T{0};
        }

        if (t * TILE_SIZE + threadIdx.y < K && global_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + global_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = T{0};
        }

        if (t * TILE_SIZE + threadIdx.y < K && global_col + blockDim.x < N) {
            tile_B[threadIdx.y][threadIdx.x + blockDim.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + global_col + blockDim.x];
        } else {
            tile_B[threadIdx.y][threadIdx.x + blockDim.x] = T{0};
        }

        __syncthreads();

        // Compute partial results
        for (size_t k = 0; k < TILE_SIZE; ++k) {
            T a0 = tile_A[threadIdx.y][k];
            T a1 = tile_A[threadIdx.y + blockDim.y][k];
            T b0 = tile_B[k][threadIdx.x];
            T b1 = tile_B[k][threadIdx.x + blockDim.x];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum00;
    }
    if (global_row + blockDim.y < M && global_col + blockDim.x < N) {
        C[(global_row + blockDim.y) * N + global_col + blockDim.x] = sum11;
    }
}

// Warp-level matrix multiplication using WMMA (Tensor Cores)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#include <mma.hpp>
#include <cuda_fp16.h>

template<typename T>
__global__ void matrix_multiply_wmma_kernel(const half* A, const half* B, float* C,
                                           size_t M, size_t N, size_t K) {
    using namespace nvcuda::wmma;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Shared memory tiles
    __shared__ half shmem_A[64][64];  // 64x64 for better bank conflict handling
    __shared__ half shmem_B[64][64];

    // Warp and lane identification
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 64;

    // WMMA fragment
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);

    for (size_t k = 0; k < K; k += WMMA_K) {
        // Load fragments from shared memory
        load_matrix_sync(a_frag, &shmem_A[0][0], 64);
        load_matrix_sync(b_frag, &shmem_B[0][0], 64);

        // Perform matrix multiplication
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the result
    store_matrix_sync(&C[warpM * WMMA_M * N + warpN * WMMA_N], 
                     acc_frag, N, mem_row_major);
}
#endif

} // namespace hpc::gpu
```

---

## Phase 8: Parallel Primitives

### Step 8.1: Reduction Kernels

**File: `include/reductions.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>

namespace hpc::gpu {

// Reduction kernel using shared memory (tree reduction)
template<typename T, unsigned int BLOCK_SIZE>
__global__ void reduction_kernel(const T* input, T* output, size_t n) {
    extern __shared__ T sdata[];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    size_t grid_size = blockDim.x * 2 * gridDim.x;

    T sum = T{0};

    // Sequential reduction in global memory
    while (i < n) {
        sum += input[i];
        if (i + blockDim.x < n) {
            sum += input[i + blockDim.x];
        }
        i += grid_size;
    }

    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    // Unroll last warp
    if (tid < 32) {
        volatile T* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Warp-level reduction using warp shuffle
template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__global__ void warp_reduction_kernel(const T* input, T* output, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    T sum = T{0};
    for (size_t i = tid; i < n; i += stride) {
        sum += input[i];
    }

    sum = warp_reduce_sum(sum);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&output[0], sum);
    }
}

// Find minimum using warp primitives
template<typename T>
__device__ T warp_reduce_min(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__global__ void min_reduction_kernel(const T* input, T* output, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    T min_val = (tid < n) ? input[tid] : std::numeric_limits<T>::max();
    for (size_t i = tid + stride; i < n; i += stride) {
        min_val = min(min_val, input[i]);
    }

    min_val = warp_reduce_min(min_val);

    __shared__ T shared_min[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        shared_min[warp_id] = min_val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        min_val = (threadIdx.x < (blockDim.x + 31) / 32) ? 
                  shared_min[threadIdx.x] : std::numeric_limits<T>::max();
        min_val = warp_reduce_min(min_val);
        if (threadIdx.x == 0) {
            atomicMin(&output[0], min_val);
        }
    }
}

} // namespace hpc::gpu
```

### Step 8.2: Scan (Prefix Sum) Kernels

**File: `include/scan.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>

namespace hpc::gpu {

// Blelloch scan algorithm (exclusive scan)
template<typename T>
__global__ void scan_blelloch_kernel(const T* input, T* output, T* block_sums, 
                                    size_t n) {
    extern __shared__ T temp[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = 1;

    // Load input into shared memory
    int global_idx = bid * (2 * blockDim.x) + tid;
    temp[2 * tid] = (2 * global_idx < n) ? input[2 * global_idx] : T{0};
    temp[2 * tid + 1] = (2 * global_idx + 1 < n) ? input[2 * global_idx + 1] : T{0};

    // Build sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear the last element
    if (tid == 0) {
        temp[2 * blockDim.x - 1] = T{0};
        if (block_sums != nullptr) {
            block_sums[bid] = temp[2 * blockDim.x - 2];  // Save block sum
        }
    }

    // Traverse down tree & build scan
    for (int d = 1; d <= blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to global memory
    if (2 * global_idx < n) {
        output[2 * global_idx] = temp[2 * tid];
    }
    if (2 * global_idx + 1 < n) {
        output[2 * global_idx + 1] = temp[2 * tid + 1];
    }
}

// Add block sums to complete the scan
template<typename T>
__global__ void add_block_sums_kernel(T* data, const T* block_sums, size_t n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = 2 * blockDim.x;

    for (int i = bid * offset + tid; i < min((bid + 1) * offset, n); i += blockDim.x) {
        data[i] += block_sums[bid];
    }
}

} // namespace hpc::gpu
```

---

## Phase 9: CUDA Streams and Events

### Step 9.1: Stream Management

**File: `include/stream_manager.hpp`**

```cpp
#pragma once
#include "gpu_wrappers.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace hpc {

class StreamManager {
public:
    StreamManager() = default;

    explicit StreamManager(size_t num_streams) 
        : streams_(num_streams), events_(num_streams) {}

    size_t size() const { return streams_.size(); }

    CUDAStream& get_stream(size_t index) {
        return streams_.at(index);
    }

    CUDAEvent& get_event(size_t index) {
        return events_.at(index);
    }

    void synchronize_all() {
        for (auto& stream : streams_) {
            stream.synchronize();
        }
    }

    template<typename Func>
    void execute_on_stream(size_t stream_idx, Func&& func) {
        auto& stream = get_stream(stream_idx);
        func(stream.get());
    }

    void record_event(size_t event_idx, size_t stream_idx) {
        events_[event_idx].record(streams_[stream_idx].get());
    }

    void wait_for_event(size_t waiting_stream_idx, size_t event_idx) {
        CUDA_CHECK(cudaStreamWaitEvent(streams_[waiting_stream_idx].get(), 
                                       events_[event_idx].get(), 0));
    }

private:
    std::vector<CUDAStream> streams_;
    std::vector<CUDAEvent> events_;
};

} // namespace hpc
```

---

## Phase 10: Advanced CUDA Features

### Step 10.1: CUDA Graphs

**File: `include/cuda_graph.hpp`**

```cpp
#pragma once
#include <cuda_runtime.h>
#include "gpu_wrappers.hpp"
#include "exceptions.hpp"

namespace hpc {

class CUDAGraph {
public:
    CUDAGraph() : graph_(nullptr), exec_(nullptr) {}

    ~CUDAGraph() {
        if (exec_ != nullptr) {
            cudaGraphExecDestroy(exec_);
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
        }
    }

    void begin_capture(cudaStream_t stream) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }

    void end_capture(cudaStream_t stream) {
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    }

    void instantiate() {
        if (graph_ == nullptr) {
            throw HPCException("Graph not captured");
        }
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
    }

    void launch(cudaStream_t stream = 0) {
        if (exec_ == nullptr) {
            throw HPCException("Graph not instantiated");
        }
        CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    }

    void update() {
        if (exec_ == nullptr || graph_ == nullptr) {
            throw HPCException("Graph not instantiated or captured");
        }
        CUDA_CHECK(cudaGraphExecUpdate(exec_, graph_, nullptr, nullptr));
    }

private:
    cudaGraph_t graph_;
    cudaGraphExec_t exec_;
};

} // namespace hpc
```

---

## Phase 11: Integration and High-Level API

### Step 11.1: Main Header

**File: `include/hpc_toolkit.hpp`**

```cpp
#pragma once
#include "matrix.hpp"
#include "vector.hpp"
#include "gpu_wrappers.hpp"
#include "gpu_kernels.hpp"
#include "optimized_kernels.hpp"
#include "reductions.hpp"
#include "scan.hpp"
#include "stream_manager.hpp"
#include "cuda_graph.hpp"
#include "utils.hpp"

namespace hpc {

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
        CUDA_CHECK(cudaMemcpy(result.data(), data_.data(), 
                             size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }

    MatrixGPU multiply(const MatrixGPU& other, bool use_optimized = true) const;

private:
    size_t rows_;
    size_t cols_;
    size_t size_;
    UnifiedMemory<T> data_;
};

template<typename T>
MatrixGPU<T> MatrixGPU<T>::multiply(const MatrixGPU& other, bool use_optimized) const {
    if (cols_ != other.rows_) {
        throw DimensionMismatchError(rows_, cols_, other.rows_, other.cols());
    }

    MatrixGPU<T> result(rows_, other.cols_);

    dim3 block(32, 32);
    dim3 grid((other.cols_ + block.x - 1) / block.x, 
              (rows_ + block.y - 1) / block.y);

    if (use_optimized) {
        gpu::matrix_multiply_tiled_kernel<T, 32><<<grid, block>>>(
            data_.data(), other.data_.data(), result.data_.data(),
            rows_, other.cols_, cols_);
    } else {
        gpu::matrix_multiply_simple_kernel<<<grid, block>>>(
            data_.data(), other.data_.data(), result.data_.data(),
            rows_, other.cols_, cols_);
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}

} // namespace hpc
```

---

## Phase 12: Example Programs

### Step 12.1: Example 1 - Basics

**File: `examples/example1_basics.cpp`**

```cpp
#include "../include/hpc_toolkit.hpp"
#include <iostream>

int main() {
    print_header("Example 1: Basic Operations");

    // Create CPU matrices
    auto A = Matrix<float>::random(4, 4);
    auto B = Matrix<float>::random(4, 4);

    A.print("Matrix A");
    B.print("Matrix B");

    // CPU matrix addition
    auto C_cpu = A + B;
    C_cpu.print("A + B (CPU)");

    // CPU matrix multiplication
    auto D_cpu = A.multiply_cpu(B);
    D_cpu.print("A * B (CPU)");

    // GPU matrix multiplication
    MatrixGPU<float> A_gpu(4, 4);
    MatrixGPU<float> B_gpu(4, 4);
    A_gpu.upload(A);
    B_gpu.upload(B);

    auto D_gpu = A_gpu.multiply(B_gpu);
    auto D_gpu_host = D_gpu.download();
    D_gpu_host.print("A * B (GPU)");

    // Verify results match
    if (D_cpu == D_gpu_host) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results differ!" << std::endl;
    }

    // Vector operations
    auto v = Vector<float>::random(8);
    v.print("Vector v");

    std::cout << "Sum: " << v.sum() << std::endl;
    std::cout << "Mean: " << v.mean() << std::endl;
    std::cout << "Norm: " << v.norm() << std::endl;

    return 0;
}
```

This completes the comprehensive implementation guide. Each phase builds upon the previous ones, creating a fully functional HPC Toolkit with modern C++ and CUDA features.
