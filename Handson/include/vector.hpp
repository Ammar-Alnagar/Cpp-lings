#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <concepts>
#include <span>
#include "utils.hpp"

namespace hpc {

template<typename T>
class Vector {
public:
    using iterator = T*;
    using const_iterator = const T*;

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

    void sort() {
        std::sort(begin(), end());
    }

    void sort_descending() {
        std::sort(begin(), end(), std::greater<T>());
    }

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
