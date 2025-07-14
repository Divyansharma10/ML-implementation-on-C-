#include "matrix_operations.hpp"

namespace ml {

// Explicit template instantiations for common types
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;

// Explicit instantiation of non-member functions
template Matrix<float> operator*(const float& scalar, const Matrix<float>& matrix);
template Matrix<double> operator*(const double& scalar, const Matrix<double>& matrix);
template Matrix<int> operator*(const int& scalar, const Matrix<int>& matrix);

template std::ostream& operator<<(std::ostream& os, const Matrix<float>& matrix);
template std::ostream& operator<<(std::ostream& os, const Matrix<double>& matrix);
template std::ostream& operator<<(std::ostream& os, const Matrix<int>& matrix);

// Explicit instantiation of utility functions
template Matrix<float> zeros<float>(size_t rows, size_t cols);
template Matrix<double> zeros<double>(size_t rows, size_t cols);
template Matrix<int> zeros<int>(size_t rows, size_t cols);

template Matrix<float> ones<float>(size_t rows, size_t cols);
template Matrix<double> ones<double>(size_t rows, size_t cols);
template Matrix<int> ones<int>(size_t rows, size_t cols);

template Matrix<float> identity<float>(size_t size);
template Matrix<double> identity<double>(size_t size);
template Matrix<int> identity<int>(size_t size);

template Matrix<float> random_matrix<float>(size_t rows, size_t cols, float min_val, float max_val);
template Matrix<double> random_matrix<double>(size_t rows, size_t cols, double min_val, double max_val);

template Matrix<float> random_normal_matrix<float>(size_t rows, size_t cols, float mean, float stddev);
template Matrix<double> random_normal_matrix<double>(size_t rows, size_t cols, double mean, double stddev);

} // namespace ml