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

    Matrix multiply_cpu(const Matrix& other) const;

    Matrix elementwise_multiply(const Matrix& other) const {
        check_dimensions(other);
        Matrix result(rows_, cols_);
        std::transform(data_.get(), data_.get() + size_, 
                       other.data_.get(), result.data_.get(), 
                       std::multiplies<T>());
        return result;
    }

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
