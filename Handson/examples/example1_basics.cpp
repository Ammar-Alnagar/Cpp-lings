#include "hpc_toolkit.hpp"
#include <iostream>

int main() {
    print_header("Example 1: Basic Operations");

    std::cout << "\n=== Matrix Operations ===" << std::endl;

    auto A = Matrix<float>::random(4, 4, -1.0f, 1.0f);
    auto B = Matrix<float>::random(4, 4, -1.0f, 1.0f);

    A.print("Matrix A");
    B.print("Matrix B");

    auto C = A + B;
    C.print("A + B");

    auto D = A.multiply_cpu(B);
    D.print("A * B (CPU)");

    std::cout << "\n=== Vector Operations ===" << std::endl;

    auto v = Vector<float>::random(8, -1.0f, 1.0f);
    v.print("Vector v");

    std::cout << "Sum: " << v.sum() << std::endl;
    std::cout << "Mean: " << v.mean() << std::endl;
    std::cout << "Norm: " << v.norm() << std::endl;
    std::cout << "Min: " << v.min() << std::endl;
    std::cout << "Max: " << v.max() << std::endl;

    auto v2 = Vector<float>::ones(8);
    auto v3 = v + v2;
    v3.print("v + ones(8)");

    std::cout << "\nExample 1 completed successfully!" << std::endl;

    return 0;
}
