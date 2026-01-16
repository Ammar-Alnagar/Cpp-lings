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
