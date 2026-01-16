# Modern C++ and CUDA Hands-On Project

A comprehensive, practical project demonstrating modern C++ (C++17/20/23) and modern CUDA programming techniques. This project implements a high-performance computing toolkit with CPU and GPU implementations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Modern C++ Concepts Demonstrated](#modern-c-concepts-demonstrated)
6. [Modern CUDA Concepts Demonstrated](#modern-cuda-concepts-demonstrated)
7. [Getting Started Guide](#getting-started-guide)
8. [Step-by-Step Implementation](#step-by-step-implementation)
9. [Running the Examples](#running-the-examples)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Troubleshooting](#troubleshooting)
12. [Further Learning](#further-learning)

## Project Overview

This project implements a High-Performance Computing (HPC) Toolkit that includes:

- **Matrix Operations**: Matrix multiplication, transposition, decomposition
- **Vector Operations**: Element-wise operations, dot products, reductions
- **Parallel Algorithms**: Sort, scan, reduce, prefix sum
- **Image Processing**: Convolution, filtering, transformations
- **Numerical Methods**: Monte Carlo simulation, linear solvers
- **GPU Acceleration**: CUDA implementations with unified memory, streams, and async operations

The toolkit provides both CPU and GPU implementations, allowing you to compare performance and understand the trade-offs.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with Compute Capability 7.0 or higher (for modern CUDA features)
- At least 4GB GPU memory
- 8GB+ system RAM recommended

### Software Requirements
- **C++ Compiler**: GCC 11+ or Clang 13+ with C++20 support
- **CUDA Toolkit**: 11.8 or 12.0+ (recommended)
- **CMake**: 3.25 or later
- **cuDNN**: 8.6 or later (optional, for advanced examples)
- **NVIDIA Driver**: 520.61.05 or later

### System Requirements
- Linux (Ubuntu 20.04/22.04 recommended)
- CUDA-capable GPU with proper drivers installed
- At least 10GB free disk space

## Installation

### Step 1: Verify CUDA Installation

```bash
nvcc --version
```

Expected output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on <date>
Cuda compilation tools, release 12.0, V12.0.xxx
```

### Step 2: Verify GPU

```bash
nvidia-smi
```

This should display your GPU information including compute capability.

### Step 3: Clone and Build

```bash
cd Handson
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Project Structure

```
Handson/
├── CMakeLists.txt              # Main CMake configuration
├── Makefile                    # Alternative build system
├── README.md                   # This file
├── IMPLEMENTATION_GUIDE.md     # Detailed step-by-step guide
├── include/                    # Header files
│   ├── hpc_toolkit.hpp       # Main header
│   ├── matrix.hpp            # Matrix operations
│   ├── vector.hpp            # Vector operations
│   ├── gpu_wrappers.hpp      # CUDA wrapper classes
│   └── utils.hpp             # Utility functions
├── src/                       # Source files
│   ├── cpu_matrix.cpp        # CPU implementations
│   ├── gpu_matrix.cu         # GPU implementations (CUDA)
│   ├── cpu_vector.cpp        # CPU vector operations
│   ├── gpu_vector.cu         # GPU vector operations
│   ├── reductions.cu         # Reduction kernels
│   └── main.cpp              # Example programs
├── examples/                  # Example programs
│   ├── example1_basics.cpp
│   ├── example2_matrix_ops.cpp
│   ├── example3_gpu_acceleration.cpp
│   ├── example4_streams.cpp
│   └── example5_advanced.cpp
├── benchmarks/                # Benchmark scripts
│   └── benchmark_suite.cpp
└── tests/                    # Unit tests
    └── test_suite.cpp
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
- Coroutines
- Modules
- Spaceship operator (<=>)
- Designated initializers
- consteval and constinit
- std::span
- Formatting library (std::format)
- std::jthread

### C++23 Features (experimental)
- std::print and std::println
- std::expected
- Deducing this
- std::mdspan
- std::generator

### Modern C++ Best Practices
- RAII and smart pointers (std::unique_ptr, std::shared_ptr)
- Move semantics and perfect forwarding
- Rule of zero/five
- Const-correctness
- No exceptions policy (where appropriate)
- Type traits and SFINAE
- Template metaprogramming
- constexpr functions

## Modern CUDA Concepts Demonstrated

### CUDA 11/12 Features
- Unified Memory Management (cudaMallocManaged)
- CUDA Streams and Events
- Asynchronous memory transfers and kernel launches
- CUDA Graphs
- Cooperative Groups
- Warp-level primitives (warp-level matrix operations)
- Shared memory optimization
- Dynamic parallelism
- Multi-GPU programming
- CUDA Memory Pool

### Advanced Techniques
- Vectorized memory loads/stores
- Thread coarsening
- Loop unrolling
- Tiling/blocking strategies
- Reduction algorithms
- Scan/prefix-sum algorithms
- Sort algorithms (thrust, cub)
- Texture and surface memory
- Constant memory optimization
- Atomics and synchronization

### Performance Optimization
- Occupancy analysis
- Register pressure management
- Shared memory bank conflicts
- Memory coalescing
- Instruction throughput optimization
- Kernel fusion
- Overlapping computation and communication

## Getting Started Guide

### Quick Start Example

```cpp
#include "hpc_toolkit.hpp"

int main() {
    // Create matrices
    auto A = Matrix<float>::random(1024, 1024);
    auto B = Matrix<float>::random(1024, 1024);
    
    // CPU computation
    auto C_cpu = Matrix<float>::multiply_cpu(A, B);
    
    // GPU computation
    auto C_gpu = Matrix<float>::multiply_gpu(A, B);
    
    // Verify results
    assert(C_cpu == C_gpu);
    
    return 0;
}
```

### Compilation

```bash
cd Handson/build
make quick_start
./examples/quick_start
```

## Step-by-Step Implementation

### Phase 1: Basic Setup and Utilities

1. **Set up the build system**
   - Configure CMake for mixed C++/CUDA compilation
   - Set up compiler flags for modern C++ (C++20)
   - Configure CUDA architecture flags

2. **Create utility classes**
   - Exception handling hierarchy
   - Logging utilities
   - Timing utilities
   - Memory management wrappers

### Phase 2: CPU Implementations

3. **Matrix class (CPU)**
   - Template-based matrix class
   - Basic operations (addition, multiplication)
   - Memory layout (row-major vs column-major)
   - Exception safety

4. **Vector class (CPU)**
   - Vector operations
   - STL integration
   - Algorithm implementations

### Phase 3: GPU Fundamentals

5. **CUDA memory management**
   - RAII wrappers for CUDA memory
   - Unified memory management
   - Error handling macros

6. **Basic GPU kernels**
   - Vector addition kernel
   - Matrix multiplication kernel
   - Thread indexing patterns

### Phase 4: Advanced GPU Techniques

7. **Optimized matrix multiplication**
   - Tiled/shared memory implementation
   - Register tiling
   - Tensor cores (if available)

8. **Parallel primitives**
   - Reduction kernel
   - Prefix sum/scan
   - Parallel sort

### Phase 5: Streams and Concurrency

9. **CUDA streams**
   - Stream management class
   - Overlapping transfers and computation
   - Multiple stream usage

10. **CUDA events**
    - Event synchronization
    - Timing and profiling

### Phase 6: Advanced Features

11. **CUDA Graphs**
    - Graph capture
    - Graph execution
    - Graph updates

12. **Cooperative Groups**
    - Thread block groups
    - Warp groups
    - Grid groups

13. **Multi-GPU support**
    - Peer-to-peer communication
    - Multi-GPU computation

### Phase 7: Integration and Examples

14. **High-level API**
    - Unified CPU/GPU interface
    - Automatic device selection
    - Performance-aware routing

15. **Example programs**
    - Basic operations
    - Performance comparisons
    - Real-world applications

## Running the Examples

### Example 1: Basics

Demonstrates basic matrix and vector operations on CPU and GPU.

```bash
./examples/example1_basics
```

### Example 2: Matrix Operations

Comprehensive matrix operations including multiplication, transposition, and decomposition.

```bash
./examples/example2_matrix_ops
```

### Example 3: GPU Acceleration

Compares CPU vs GPU performance for various operations.

```bash
./examples/example3_gpu_acceleration
```

### Example 4: Streams and Concurrency

Demonstrates CUDA streams and asynchronous operations.

```bash
./examples/example4_streams
```

### Example 5: Advanced Features

Shows CUDA graphs, cooperative groups, and advanced optimizations.

```bash
./examples/example5_advanced
```

## Performance Benchmarks

Run the comprehensive benchmark suite:

```bash
./benchmarks/benchmark_suite
```

This will generate a report comparing CPU and GPU performance across:
- Matrix multiplication (various sizes)
- Vector reductions
- Parallel sort
- Image processing operations

Expected performance (RTX 3090, Intel i9-12900K):
- 1024x1024 Matrix Mul: CPU ~50ms, GPU ~0.5ms (100x speedup)
- 1M Element Reduction: CPU ~8ms, GPU ~0.3ms (26x speedup)
- 10M Element Sort: CPU ~800ms, GPU ~12ms (66x speedup)

## Troubleshooting

### CUDA Out of Memory

Reduce problem size or check GPU memory usage:
```bash
nvidia-smi
```

### Compilation Errors

Ensure your CUDA toolkit is properly installed:
```bash
echo $CUDA_HOME
nvcc --version
```

### Kernel Launch Failures

Check kernel launch errors:
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}
```

### Performance Issues

Use NVIDIA Nsight for profiling:
```bash
nsys profile --stats=true ./your_program
```

## Further Learning

### Resources
- CUDA C++ Programming Guide
- Modern C++ Design (Andrei Alexandrescu)
- C++ Concurrency in Action (Anthony Williams)
- NVIDIA Developer Blog
- GPU Gems series

### Advanced Topics
- Tensor cores for deep learning
- CUDA libraries (cuBLAS, cuDNN, cuTENSOR)
- Mixed precision computation
- Tensor cores and WMMA API
- CUDA Graph optimizations
- Persistent kernels

### Project Extensions
- Add support for sparse matrices
- Implement deep learning primitives
- Create Python bindings (PyBind11)
- Add distributed computing support
- Implement automatic kernel tuning
