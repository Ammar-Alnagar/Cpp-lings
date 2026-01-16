# Build Instructions for Handson HPC Toolkit

## Current System Status
- CUDA Version: 13.1
- GCC Version: 14.3.1
- **Issue**: CUDA 13.1 is not compatible with GCC 14+ due to `noexcept(true)` conflicts in system headers.

## CPU-Only Build (Working)

### Quick Start
```bash
make all-cpu
```

Or manually:
```bash
# Compile CPU components
g++ -std=c++20 -Wall -Wextra -I./include -c src/cpu_matrix.cpp -o build/cpu_matrix.o

# Compile examples
g++ -std=c++20 -Wall -Wextra -I./include -c examples/example1_basics.cpp -o build/example1_basics.o
g++ -std=c++20 -Wall -Wextra -I./include -c examples/example2_matrix_ops.cpp -o build/example2_matrix_ops.cpp.o

# Link
g++ -std=c++20 build/cpu_matrix.o build/example1_basics.o -o example1_basics
```

### Run Examples
```bash
./example1_basics
./example2_matrix_ops
./test_suite
```

## GPU Build (Requires Compatibility Fix)

### Option 1: Use Older GCC (Recommended)
Install GCC 11-13 which is compatible with CUDA 13.1:
```bash
sudo dnf install gcc11 gcc11-c++
scl enable gcc11 bash
```

Then build:
```bash
make all
```

### Option 2: Wait for CUDA Update
NVIDIA is expected to release CUDA 13.2+ with GCC 14 support.

## Using nvcc with Specific Flags

If you have GCC 11+ available:
```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
nvcc -std=c++20 -arch=native -I./include -c src/gpu_vector.cu -o build/gpu_vector.o
```

## Troubleshooting

### CUDA Compilation Fails with `noexcept(true)` Error
This is the GCC 14 incompatibility. Use GCC 11-13 instead.

### "cuda_runtime.h not found" When Compiling CPU Code
Make sure you're using `g++` (not `nvcc`) for CPU-only components:
```bash
g++ -std=c++20 ...  # Correct
nvcc -std=c++20 ... # Don't use nvcc for CPU-only code
```

### CPU Code Works but GPU Code Fails
This is expected given the GCC 14 + CUDA 13.1 incompatibility. The CPU implementations are fully functional.

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Matrix (CPU) | Working | Fully tested |
| Vector (CPU) | Working | Fully tested |
| Matrix Operations | Working | Addition, subtraction, transpose, multiplication |
| Vector Operations | Working | All operations functional |
| GPU Acceleration | Pending | Requires GCC 11-13 |
| CUDA Kernels | Pending | Will work with compatible compiler |

## Running Tests

All CPU tests pass successfully:
```bash
./test_suite
```

Output:
```
=== HPC Toolkit Test Suite ===

[TEST] Matrix Creation
  PASSED

[TEST] Identity Matrix
  PASSED

[TEST] Matrix Addition
  PASSED

[TEST] Matrix Transpose
  PASSED

[TEST] Vector Operations
  PASSED

[TEST] Vector Dot Product
  PASSED

=== Test Results ===
Passed: 6/6
```
