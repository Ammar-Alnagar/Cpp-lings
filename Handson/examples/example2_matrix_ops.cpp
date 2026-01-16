#include "hpc_toolkit.hpp"
#include <iostream>

int main() {
    print_header("Example 2: Matrix Operations");

    std::cout << "\n=== Creating Identity Matrix ===" << std::endl;
    auto I = Matrix<float>::identity(5);
    I.print("5x5 Identity Matrix");

    std::cout << "\n=== Matrix Transposition ===" << std::endl;
    auto A = Matrix<float>::random(3, 4, 0.0f, 10.0f);
    A.print("Matrix A (3x4)");
    auto A_T = A.transpose();
    A_T.print("Transpose of A (4x3)");

    std::cout << "\n=== Element-wise Operations ===" << std::endl;
    auto B = Matrix<float>::random(3, 4, 0.0f, 5.0f);
    auto C = A.elementwise_multiply(B);
    A.print("Matrix A");
    B.print("Matrix B");
    C.print("A .* B (Element-wise multiply)");

    std::cout << "\n=== Scalar Multiplication ===" << std::endl;
    auto D = A * 2.5f;
    A.print("Matrix A");
    D.print("A * 2.5");

    std::cout << "\n=== Matrix Multiplication CPU ===" << std::endl;
    auto M1 = Matrix<float>::random(256, 256, -1.0f, 1.0f);
    auto M2 = Matrix<float>::random(256, 256, -1.0f, 1.0f);

    std::cout << "M1: " << M1.rows() << "x" << M1.cols() << std::endl;
    std::cout << "M2: " << M2.rows() << "x" << M2.cols() << std::endl;

    {
        ScopedTimer timer("CPU Matrix Multiplication");
        auto M3 = M1.multiply_cpu(M2);
        std::cout << "Result M3: " << M3.rows() << "x" << M3.cols() << std::endl;
    }

    std::cout << "\nExample 2 completed successfully!" << std::endl;

    return 0;
}
