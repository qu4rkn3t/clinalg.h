#include "clinalg.h"
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

void run_api_demo() {
  std::cout << "===============================================" << '\n';
  std::cout << "           Linear Algebra API Demo" << '\n';
  std::cout << "===============================================" << '\n';

  lin::Matrix<double> I = lin::Matrix<double>::identity(4);
  I.print("I (4x4 Identity):");

  lin::Matrix<double> A = lin::Matrix<double>::random(4, 4);
  A.print("A (4x4 Random):");

  lin::Matrix<double> B = 2.0 * A;
  B.print("B = 2.0 * A:");

  lin::Matrix<double> C = A + I;
  C.print("C = A + I:");

  lin::Matrix<double> D = A * C;
  D.print("D = A * C (Optimized GEMM):");

  lin::Vector<double> v({1.0, 2.0, 3.0, 4.0});
  v.print("v (Vector):");

  lin::Vector<double> w = D * v;
  w.print("w = D * v:");

  double dot_prod = v.dot(w);
  std::cout << "Dot Product (v . w): " << dot_prod << '\n';
}

void run_benchmark() {
  std::cout << "===============================================" << '\n';
  std::cout << "         Optimized GEMM Benchmark" << '\n';
  std::cout << "===============================================" << '\n';

  const int N = 1024;

  std::cout << "Matrix Size (N): " << N << '\n';
  std::cout << "Using " << std::thread::hardware_concurrency() << " hardware threads." << '\n';
  std::cout << "Datatype: double" << '\n';

  std::cout << "Generating random matrices..." << '\n';
  lin::Matrix<double> A = lin::Matrix<double>::random(N, N);
  lin::Matrix<double> B = lin::Matrix<double>::random(N, N);

  std::cout << "Running benchmark..." << '\n';
  TimePoint start = Clock::now();
  
  lin::Matrix<double> C = A * B;
  
  TimePoint end = Clock::now();
  double time_gemm = Duration(end - start).count();

  std::cout << "-----------------------------------------------" << '\n';
  std::cout << "Optimized Parallel GEMM: " << time_gemm << " seconds" << '\n';
  std::cout << "===============================================" << '\n';
  
  if (C.rows() == N) {
    std::cout << "Benchmark complete. (Result C(0,0) = " << C(0,0) << ")" << '\n';
  } else {
    std::cout << "Benchmark FAILED." << '\n';
  }
}


int main() {
  try {
    run_api_demo();
    run_benchmark();
  } catch (const std::exception& e) {
    std::cerr << "An error occurred: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
