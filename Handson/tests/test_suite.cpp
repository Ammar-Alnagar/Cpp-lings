#include "hpc_toolkit.hpp"
#include <iostream>
#include <cassert>

using namespace hpc;

int main() {
    std::cout << "=== HPC Toolkit Test Suite ===" << std::endl;

    int passed = 0;
    int total = 0;

    std::cout << "\n[TEST] Matrix Creation" << std::endl;
    total++;
    try {
        auto m = Matrix<float>::zeros(3, 3);
        assert(m.rows() == 3);
        assert(m.cols() == 3);
        passed++;
        std::cout << "  PASSED" << std::endl;
    } catch (...) {
        std::cout << "  FAILED" << std::endl;
    }

    std::cout << "\n[TEST] Identity Matrix" << std::endl;
    total++;
    try {
        auto I = Matrix<float>::identity(4);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                float expected = (i == j) ? 1.0f : 0.0f;
                assert(std::abs(I(i, j) - expected) < 1e-6f);
            }
        }
        passed++;
        std::cout << "  PASSED" << std::endl;
    } catch (...) {
        std::cout << "  FAILED" << std::endl;
    }

    std::cout << "\n[TEST] Matrix Addition" << std::endl;
    total++;
    try {
        auto A = Matrix<float>::ones(2, 2);
        auto B = Matrix<float>::ones(2, 2);
        auto C = A + B;
        for (size_t i = 0; i < 4; ++i) {
            assert(std::abs(C.data()[i] - 2.0f) < 1e-6f);
        }
        passed++;
        std::cout << "  PASSED" << std::endl;
    } catch (...) {
        std::cout << "  FAILED" << std::endl;
    }

    std::cout << "\n[TEST] Matrix Transpose" << std::endl;
    total++;
    try {
        auto A = Matrix<float>::random(3, 4);
        auto A_T = A.transpose();
        assert(A_T.rows() == 4);
        assert(A_T.cols() == 3);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                assert(std::abs(A(i, j) - A_T(j, i)) < 1e-6f);
            }
        }
        passed++;
        std::cout << "  PASSED" << std::endl;
    } catch (...) {
        std::cout << "  FAILED" << std::endl;
    }

    std::cout << "\n[TEST] Vector Operations" << std::endl;
    total++;
    try {
        auto v = Vector<float>::ones(10);
        assert(v.size() == 10);
        assert(std::abs(v.sum() - 10.0f) < 1e-6f);
        assert(std::abs(v.mean() - 1.0f) < 1e-6f);
        passed++;
        std::cout << "  PASSED" << std::endl;
    } catch (...) {
        std::cout << "  FAILED" << std::endl;
    }

    std::cout << "\n[TEST] Vector Dot Product" << std::endl;
    total++;
    try {
        auto v1 = Vector<float>::ones(5);
        auto v2 = Vector<float>::ones(5);
        float dot = v1.dot(v2);
        assert(std::abs(dot - 5.0f) < 1e-6f);
        passed++;
        std::cout << "  PASSED" << std::endl;
    } catch (...) {
        std::cout << "  FAILED" << std::endl;
    }

    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;

    return (passed == total) ? 0 : 1;
}
