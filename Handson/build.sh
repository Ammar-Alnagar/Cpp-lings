#!/bin/bash

# Build script for Handson HPC Toolkit (CPU-only version)

set -e

echo "========================================="
echo "Building Handson HPC Toolkit"
echo "========================================="
echo ""

# Create build directory
mkdir -p build

# Compile CPU matrix implementation
echo "[1/4] Compiling CPU matrix implementation..."
g++ -std=c++20 -Wall -Wextra -I./include -c src/cpu_matrix.cpp -o build/cpu_matrix.o

# Compile example1_basics
echo "[2/4] Compiling example1_basics..."
g++ -std=c++20 -Wall -Wextra -I./include -c examples/example1_basics.cpp -o build/example1_basics.o
g++ -std=c++20 build/cpu_matrix.o build/example1_basics.o -o example1_basics

# Compile example2_matrix_ops
echo "[3/4] Compiling example2_matrix_ops..."
g++ -std=c++20 -Wall -Wextra -I./include -c examples/example2_matrix_ops.cpp -o build/example2_matrix_ops.o
g++ -std=c++20 build/cpu_matrix.o build/example2_matrix_ops.o -o example2_matrix_ops

# Compile test suite
echo "[4/4] Compiling test suite..."
g++ -std=c++20 -Wall -Wextra -I./include -c tests/test_suite.cpp -o build/test_suite.o
g++ -std=c++20 build/cpu_matrix.o build/test_suite.o -o test_suite

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo ""
echo "Available executables:"
echo "  ./example1_basics     - Basic matrix and vector operations"
echo "  ./example2_matrix_ops - Advanced matrix operations"
echo "  ./test_suite          - Run test suite"
echo ""
echo "Run ./test_suite to verify the build."
