#include "matrix.hpp"
#include <algorithm>
#include <stdexcept>

namespace hpc {

template<typename T>
Matrix<T> Matrix<T>::multiply_cpu(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw DimensionMismatchError(rows_, cols_, other.rows_, other.cols_);
    }

    Matrix result(rows_, other.cols_, T{0});

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = 0; k < cols_; ++k) {
            T a_ik = (*this)(i, k);
            for (size_t j = 0; j < other.cols_; ++j) {
                result(i, j) += a_ik * other(k, j);
            }
        }
    }

    return result;
}

template class Matrix<float>;
template class Matrix<double>;

} // namespace hpc
