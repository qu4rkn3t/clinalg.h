#pragma once

#include <iostream>
#include <vector>
#include <iomanip> 
#include <stdexcept>
#include <string>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <numeric>
#include <algorithm>
#include <cassert>

namespace lin {

template <typename T>
class Matrix;

template <typename T>
class Vector {
  size_t size_;
  std::vector<T> data_;

public:
  Vector() : size_(0) {}
  explicit Vector(size_t size) : size_(size), data_(size) {}
  Vector(size_t size, T init_val) : size_(size), data_(size, init_val) {}
  
  Vector(std::initializer_list<T> list) : size_(list.size()), data_(list) {}

  T& operator()(size_t i) {
    assert(i < size_ && "Vector access out of bounds");
    return data_[i];
  }
  const T& operator()(size_t i) const {
    assert(i < size_ && "Vector access out of bounds");
    return data_[i];
  }

  size_t size() const { return size_; }

  void print(const std::string& title = "") const {
    if (!title.empty()) {
      std::cout << title << "\n";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < size_; ++i) {
      std::cout << std::setw(10) << data_[i] << "\n";
    }
    std::cout << "\n";
  }

  T dot(const Vector<T>& other) const {
    if (size_ != other.size_) {
      throw std::runtime_error("Vector dot product: size mismatch.");
    }
    T sum = 0;
    for (size_t i = 0; i < size_; ++i) {
      sum += data_[i] * other.data_[i];
    }
    return sum;
  }
};


template <typename T>
class Matrix {
  size_t rows_, cols_;
  std::vector<T> data_;

public:
  Matrix() : rows_(0), cols_(0) {}
  Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
  Matrix(size_t rows, size_t cols, T init_val) : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

  Matrix(const Matrix& other) = default;
  Matrix& operator=(const Matrix& other) = default;

  Matrix(Matrix&& other) noexcept = default;
  Matrix& operator=(Matrix&& other) noexcept = default;

  static Matrix identity(size_t n) {
    Matrix m(n, n, 0);
    for (size_t i = 0; i < n; ++i) {
      m(i, i) = 1;
    }
    return m;
  }
  static Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0);
  }
  static Matrix random(size_t rows, size_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        m(i, j) = static_cast<T>(dis(gen));
      }
    }
    return m;
  }

  T& operator()(size_t r, size_t c) {
    assert(r < rows_ && c < cols_ && "Matrix access out of bounds");
    return data_[r * cols_ + c];
  }
  const T& operator()(size_t r, size_t c) const {
    assert(r < rows_ && c < cols_ && "Matrix access out of bounds");
    return data_[r * cols_ + c];
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  void print(const std::string& title = "") const {
    if (!title.empty()) {
      std::cout << title << "\n";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        std::cout << std::setw(10) << (*this)(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  Matrix transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }
};


template <typename T>
Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::runtime_error("Matrix addition: dimension mismatch.");
  }
  Matrix<T> C(A.rows(), A.cols());
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
  return C;
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::runtime_error("Matrix subtraction: dimension mismatch.");
  }
  Matrix<T> C(A.rows(), A.cols());
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      C(i, j) = A(i, j) - B(i, j);
    }
  }
  return C;
}

template <typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& A) {
  Matrix<T> C(A.rows(), A.cols());
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      C(i, j) = scalar * A(i, j);
    }
  }
  return C;
}
template <typename T>
Matrix<T> operator*(const Matrix<T>& A, T scalar) {
  return scalar * A;
}


namespace internal {
const int TILE_SIZE = 32;

template <typename T>
void gemm_worker(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, size_t row_start, size_t row_end) {
  size_t N = A.rows();
  size_t M = B.cols();
  size_t K = A.cols();

  for (size_t i0 = row_start; i0 < row_end; i0 += TILE_SIZE) {
    for (size_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
      for (size_t j0 = 0; j0 < M; j0 += TILE_SIZE) {
        for (size_t i = i0; i < std::min(i0 + TILE_SIZE, row_end); ++i) {
          for (size_t k = k0; k < std::min(k0 + TILE_SIZE, K); ++k) {
            T r = A(i, k);
            for (size_t j = j0; j < std::min(j0 + TILE_SIZE, M); ++j) {
              C(i, j) += r * B(k, j);
            }
          }
        }
      }
    }
  }
}
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
  if (A.cols() != B.rows()) {
    throw std::runtime_error("Matrix multiplication: dimension mismatch.");
  }

  size_t N = A.rows();
  size_t M = B.cols();
  Matrix<T> C = Matrix<T>::zeros(N, M);

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;
  
  std::vector<std::future<void>> futures;
  
  size_t rows_per_thread = N / num_threads;
  size_t current_row = 0;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t row_end = (t == num_threads - 1) ? N : (current_row + rows_per_thread);

    futures.push_back(
      std::async(std::launch::async, internal::gemm_worker<T>, std::cref(A), std::cref(B), std::ref(C), current_row, row_end)
    );
    current_row = row_end;
  }
  
  for (auto& f : futures) {
    f.get();
  }

  return C;
}


template <typename T>
Vector<T> operator*(const Matrix<T>& A, const Vector<T>& v) {
  if (A.cols() != v.size()) {
    throw std::runtime_error("Matrix-Vector multiplication: dimension mismatch.");
  }
  Vector<T> C(A.rows(), 0);
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      C(i) += A(i, j) * v(j);
    }
  }
  return C;
}

}
