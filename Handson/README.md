# Modern C++ and CUDA Hands-On Project

A comprehensive, practical project demonstrating modern C++ (C++17/20/23) and modern CUDA programming techniques. This project implements a high-performance computing toolkit with CPU and GPU implementations.

## Table of Contents

1. [Current Status](#current-status)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Modern C++ Concepts Demonstrated](#modern-c-concepts-demonstrated)
5. [Modern CUDA Concepts](#modern-cuda-concepts)
6. [Building the Project](#building-the-project)
7. [Running the Examples](#running-the-examples)
8. [API Documentation](#api-documentation)
9. [Known Issues](#known-issues)

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Matrix (CPU) | Working | Fully functional and tested |
| Vector (CPU) | Working | All operations working |
| Matrix Operations | Working | Addition, subtraction, transpose, multiplication |
| Vector Operations | Working | All operations tested |
| GPU Acceleration | Pending | Requires GCC 11-13 (see Known Issues) |
| CUDA Kernels | Pending | Code written, needs compatible compiler |

## Quick Start

### CPU-Only Build (Works Now)

```bash
cd Handson

# Build all CPU components
g++ -std=c++20 -Wall -Wextra -I./include -c src/cpu_matrix.cpp -o build/cpu_matrix.o
g++ -std=c++20 -Wall -Wextra -I./include -c examples/example1_basics.cpp -o build/example1_basics.o
g++ -std=c++20 build/cpu_matrix.o build/example1_basics.o -o example1_basics

# Run example
./example1_basics

# Run tests
g++ -std=c++20 -Wall -Wextra -I./include -c tests/test_suite.cpp -o build/test_suite.o
g++ -std=c++20 build/cpu_matrix.o build/test_suite.o -o test_suite
./test_suite
```

### Example Output
```
================================================================================
Example 1: Basic Operations
================================================================================

=== Matrix Operations ===
Matrix 4x4:
[   -0.1260     0.9025    -0.7251    -0.4026]
[   -0.7793     0.6191     0.4597     0.4116]
[    0.8840     0.5196    -0.4739     0.0323]
[    0.4196     0.3103    -0.2992     0.1777]

=== Vector Operations ===
Vector v [8]: [-0.7622, -0.1598, -0.3943, 0.6719, 0.5828, -0.4700, 0.8489, 0.2720]
Sum: 0.5894
Mean: 0.0737
Norm: 1.6027
Min: -0.7622
Max: 0.8489

Example 1 completed successfully!
```

## Project Structure

```
Handson/
├── CMakeLists.txt              # CMake build configuration
├── Makefile                    # Alternative build system
├── BUILD.md                   # Detailed build instructions
├── README.md                   # This file
├── IMPLEMENTATION_GUIDE.md     # Detailed step-by-step guide
├── SUMMARY.md                 # Project summary
├── include/                    # Header files
│   ├── exceptions.hpp         # Exception hierarchy (CPU only)
│   ├── utils.hpp              # Utility functions and timers
│   ├── matrix.hpp             # Matrix class template
│   ├── vector.hpp             # Vector class template
│   ├── gpu_kernels.hpp        # Basic CUDA kernels
│   ├── optimized_kernels.hpp  # Optimized CUDA kernels
│   ├── gpu_wrappers.hpp       # RAII CUDA memory wrappers
│   ├── stream_manager.hpp      # CUDA stream management
│   └── hpc_toolkit.hpp       # Main toolkit header
├── src/                       # Source implementations
│   ├── cpu_matrix.cpp         # CPU matrix operations
│   └── gpu_vector.cu          # GPU vector operations
├── examples/                  # Example programs
│   ├── example1_basics.cpp
│   ├── example2_matrix_ops.cpp
│   └── example3_gpu_acceleration.cpp
├── tests/                    # Unit tests
│   └── test_suite.cpp
└── build/                    # Build directory (created)
```

## Modern C++ Concepts Demonstrated

### C++17 Features
- Structured bindings
- std::optional and std::variant
- std::any
- std::filesystem
- Parallel algorithms (std::execution)
- Inline variables
- If constexpr
- Fold expressions
- std::string_view

### C++20 Features
- Concepts and requires
- Ranges library
- Constexpr improvements
- Three-way comparison operator ( spaceship)
- Designated initializers
- std::span
- Formatting library

### Modern C++ Best Practices
- RAII and smart pointers (std::unique_ptr, std::shared_ptr)
- Move semantics and perfect forwarding
- Rule of zero/five
- Const-correctness
- Type traits and SFINAE
- Template metaprogramming
- Exception safety

## Modern CUDA Concepts

**Note: GPU code requires GCC 11-13. See [BUILD.md](BUILD.md) for details.**

### CUDA 11/12/13 Features
- Unified Memory (cudaMallocManaged)
- CUDA Streams and Events
- Asynchronous memory transfers and kernel launches
- CUDA Graphs
- Cooperative Groups
- Warp-level primitives
- Shared memory optimization
- Multi-GPU programming

### Advanced Techniques
- Vectorized memory loads/stores
- Thread coarsening
- Loop unrolling
- Tiling/blocking strategies
- Reduction algorithms
- Scan/prefix-sum algorithms

## Building the Project

### Prerequisites
- CUDA Toolkit 11.8+ (GPU code) 
- C++ compiler with C++20 support
- GCC 11-13 for GPU code (GCC 14+ has compatibility issues with CUDA 13.1)

### Quick Build Commands

```bash
# CPU-only (works now)
make all-cpu

# Or with CUDA (requires GCC 11-13)
make all
```

For detailed build instructions, see [BUILD.md](BUILD.md).

## Running the Examples

### Example 1: Basics
```bash
./example1_basics
```
Demonstrates basic matrix and vector operations.

### Example 2: Matrix Operations
```bash
./example2_matrix_ops
```
Shows advanced matrix operations including transposition and multiplication.

### Example 3: GPU Acceleration
```bash
./example3_gpu_acceleration
```
Compares CPU vs GPU performance (requires compatible compiler).

### Running Tests
```bash
./test_suite
```

All CPU tests pass successfully.

## API Documentation

### Matrix Operations

```cpp
#include "matrix.hpp"

using namespace hpc;

// Create matrices
auto A = Matrix<float>::random(1024, 1024);
auto B = Matrix<float>::ones(1024, 1024);
auto I = Matrix<float>::identity(4);

// Operations
auto C = A + B;              // Addition
auto D = A - B;              // Subtraction
auto E = A * 2.5f;           // Scalar multiplication
auto F = A.transpose();       // Transpose
auto G = A.multiply_cpu(B);   // Matrix multiplication

// Element-wise
auto H = A.elementwise_multiply(B);
```

### Vector Operations

```cpp
#include "vector.hpp"

using namespace hpc;

// Create vectors
auto v = Vector<float>::random(1000);
auto ones = Vector<float>::ones(10);
auto zeros = Vector<float>::zeros(5);

// Operations
auto w = v + ones;          // Addition
float sum = v.sum();        // Sum of elements
float mean = v.mean();      // Mean
float norm = v.norm();      // Euclidean norm
float dot = v.dot(w);       // Dot product
float min = v.min();        // Minimum element
float max = v.max();        // Maximum element
```

### GPU Operations (When Compatible)

```cpp
#include "hpc_toolkit.hpp"

using namespace hpc;

MatrixGPU<float> A_gpu(1024, 1024);
A_gpu.upload(A);

MatrixGPU<float> B_gpu(1024, 1024);
B_gpu.upload(B);

auto C_gpu = A_gpu.multiply(B_gpu);
auto C = C_gpu.download();
```

## Known Issues

### CUDA 13.1 + GCC 14 Incompatibility

**Problem**: CUDA 13.1 is not compatible with GCC 14 due to `noexcept(true)` conflicts in system headers.

**Symptoms**:
```
error: exception specification is incompatible with that of previous function "rsqrt"
```

**Solutions**:
1. **Use GCC 11-13** (Recommended)
   ```bash
   sudo dnf install gcc11 gcc11-c++
   scl enable gcc11 bash
   ```

2. **Wait for CUDA Update**
   - NVIDIA is expected to release CUDA 13.2+ with GCC 14 support

3. **CPU-Only Mode**
   - The CPU implementations are fully functional
   - All tests pass
   - Complete API available

For detailed troubleshooting, see [BUILD.md](BUILD.md).

## Learning Resources

### Files to Study
1. `matrix.hpp` - Modern C++ template class design
2. `vector.hpp` - Vector operations and algorithms
3. `cpu_matrix.cpp` - Efficient CPU matrix multiplication
4. `gpu_kernels.hpp` - CUDA kernel patterns
5. `optimized_kernels.hpp` - Advanced GPU optimizations
6. `exceptions.hpp` - Exception hierarchy
7. `utils.hpp` - Utility functions and timers

### Implementation Guide
For a detailed step-by-step guide, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## Performance Characteristics

Expected performance (RTX 4060, Intel CPU):

| Operation | Size | CPU Time | Expected GPU Time |
|-----------|------|----------|-------------------|
| Matrix Mul (1024) | 1024x1024 | ~50ms | ~0.5ms (100x) |
| Vector Add (1M) | 1,000,000 | ~8ms | ~0.3ms (26x) |
| Vector Reduce (10M) | 10,000,000 | ~15ms | ~0.5ms (30x) |

## License

This project is provided as an educational resource.
