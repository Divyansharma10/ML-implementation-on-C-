#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <functional>
#include <cmath>
#include <iomanip>

namespace ml {

template<typename T = double>
class Matrix {
private:
    std::vector<std::vector<T>> data_;
    size_t rows_;
    size_t cols_;

public:
    // Constructors
    Matrix() : rows_(0), cols_(0) {}
    
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_.resize(rows_, std::vector<T>(cols_, T{}));
    }
    
    Matrix(size_t rows, size_t cols, const T& value) : rows_(rows), cols_(cols) {
        data_.resize(rows_, std::vector<T>(cols_, value));
    }
    
    Matrix(const std::vector<std::vector<T>>& data) : data_(data) {
        rows_ = data_.size();
        cols_ = (rows_ > 0) ? data_[0].size() : 0;
    }
    
    Matrix(std::initializer_list<std::initializer_list<T>> init_list) {
        rows_ = init_list.size();
        cols_ = (rows_ > 0) ? init_list.begin()->size() : 0;
        data_.reserve(rows_);
        
        for (const auto& row : init_list) {
            if (row.size() != cols_) {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            data_.emplace_back(row);
        }
    }
    
    // Copy constructor
    Matrix(const Matrix& other) : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}
    
    // Move constructor
    Matrix(Matrix&& other) noexcept : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    // Assignment operators
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
        }
        return *this;
    }
    
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }
    
    // Accessors
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    bool empty() const { return rows_ == 0 || cols_ == 0; }
    
    // Element access
    T& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[row][col];
    }
    
    const T& operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[row][col];
    }
    
    std::vector<T>& operator[](size_t row) {
        if (row >= rows_) {
            throw std::out_of_range("Matrix row index out of bounds");
        }
        return data_[row];
    }
    
    const std::vector<T>& operator[](size_t row) const {
        if (row >= rows_) {
            throw std::out_of_range("Matrix row index out of bounds");
        }
        return data_[row];
    }
    
    // Matrix operations
    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] + other.data_[i][j];
            }
        }
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] - other.data_[i][j];
            }
        }
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }
        
        Matrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                T sum = T{};
                for (size_t k = 0; k < cols_; ++k) {
                    sum += data_[i][k] * other.data_[k][j];
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    Matrix operator*(const T& scalar) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] * scalar;
            }
        }
        return result;
    }
    
    Matrix operator/(const T& scalar) const {
        if (scalar == T{}) {
            throw std::invalid_argument("Division by zero");
        }
        
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] / scalar;
            }
        }
        return result;
    }
    
    // Compound assignment operators
    Matrix& operator+=(const Matrix& other) {
        *this = *this + other;
        return *this;
    }
    
    Matrix& operator-=(const Matrix& other) {
        *this = *this - other;
        return *this;
    }
    
    Matrix& operator*=(const T& scalar) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                data_[i][j] *= scalar;
            }
        }
        return *this;
    }
    
    Matrix& operator/=(const T& scalar) {
        if (scalar == T{}) {
            throw std::invalid_argument("Division by zero");
        }
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                data_[i][j] /= scalar;
            }
        }
        return *this;
    }
    
    // Comparison operators
    bool operator==(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if (data_[i][j] != other.data_[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }
    
    // Matrix-specific operations
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = data_[i][j];
            }
        }
        return result;
    }
    
    // Element-wise operations
    Matrix hadamard(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
        }
        
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[i][j] * other.data_[i][j];
            }
        }
        return result;
    }
    
    Matrix apply(std::function<T(T)> func) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = func(data_[i][j]);
            }
        }
        return result;
    }
    
    // Utility functions
    void fill(const T& value) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                data_[i][j] = value;
            }
        }
    }
    
    void zeros() {
        fill(T{});
    }
    
    void ones() {
        fill(T{1});
    }
    
    void random(T min_val = T{-1}, T max_val = T{1}) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min_val, max_val);
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                data_[i][j] = dis(gen);
            }
        }
    }
    
    void random_normal(T mean = T{}, T stddev = T{1}) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dis(mean, stddev);
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                data_[i][j] = dis(gen);
            }
        }
    }
    
    T sum() const {
        T total = T{};
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                total += data_[i][j];
            }
        }
        return total;
    }
    
    T mean() const {
        return sum() / static_cast<T>(rows_ * cols_);
    }
    
    T max() const {
        if (empty()) {
            throw std::runtime_error("Cannot find max of empty matrix");
        }
        
        T max_val = data_[0][0];
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                max_val = std::max(max_val, data_[i][j]);
            }
        }
        return max_val;
    }
    
    T min() const {
        if (empty()) {
            throw std::runtime_error("Cannot find min of empty matrix");
        }
        
        T min_val = data_[0][0];
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                min_val = std::min(min_val, data_[i][j]);
            }
        }
        return min_val;
    }
    
    void resize(size_t new_rows, size_t new_cols) {
        data_.resize(new_rows);
        for (size_t i = 0; i < new_rows; ++i) {
            data_[i].resize(new_cols, T{});
        }
        rows_ = new_rows;
        cols_ = new_cols;
    }
    
    // Row and column operations
    std::vector<T> get_row(size_t row) const {
        if (row >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        return data_[row];
    }
    
    std::vector<T> get_col(size_t col) const {
        if (col >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        
        std::vector<T> column(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            column[i] = data_[i][col];
        }
        return column;
    }
    
    void set_row(size_t row, const std::vector<T>& values) {
        if (row >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        if (values.size() != cols_) {
            throw std::invalid_argument("Row size must match matrix columns");
        }
        data_[row] = values;
    }
    
    void set_col(size_t col, const std::vector<T>& values) {
        if (col >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        if (values.size() != rows_) {
            throw std::invalid_argument("Column size must match matrix rows");
        }
        
        for (size_t i = 0; i < rows_; ++i) {
            data_[i][col] = values[i];
        }
    }
    
    // Debug and display
    void print() const {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << data_[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Iterator support
    class iterator {
    private:
        Matrix* matrix_;
        size_t row_, col_;
        
    public:
        iterator(Matrix* matrix, size_t row, size_t col) : matrix_(matrix), row_(row), col_(col) {}
        
        T& operator*() { return (*matrix_)(row_, col_); }
        T* operator->() { return &(*matrix_)(row_, col_); }
        
        iterator& operator++() {
            ++col_;
            if (col_ >= matrix_->cols_) {
                col_ = 0;
                ++row_;
            }
            return *this;
        }
        
        bool operator==(const iterator& other) const {
            return row_ == other.row_ && col_ == other.col_;
        }
        
        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }
    };
    
    iterator begin() { return iterator(this, 0, 0); }
    iterator end() { return iterator(this, rows_, 0); }
};

// Non-member operators
template<typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& matrix) {
    return matrix * scalar;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            os << std::setw(10) << std::fixed << std::setprecision(4) << matrix(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

// Utility functions for creating special matrices
template<typename T = double>
Matrix<T> zeros(size_t rows, size_t cols) {
    return Matrix<T>(rows, cols, T{});
}

template<typename T = double>
Matrix<T> ones(size_t rows, size_t cols) {
    return Matrix<T>(rows, cols, T{1});
}

template<typename T = double>
Matrix<T> identity(size_t size) {
    Matrix<T> result(size, size);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

template<typename T = double>
Matrix<T> random_matrix(size_t rows, size_t cols, T min_val = T{-1}, T max_val = T{1}) {
    Matrix<T> result(rows, cols);
    result.random(min_val, max_val);
    return result;
}

template<typename T = double>
Matrix<T> random_normal_matrix(size_t rows, size_t cols, T mean = T{}, T stddev = T{1}) {
    Matrix<T> result(rows, cols);
    result.random_normal(mean, stddev);
    return result;
}

} // namespace ml

#endif // MATRIX_OPERATIONS_HPP