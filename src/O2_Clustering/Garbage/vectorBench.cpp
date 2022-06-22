/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <vector>
#include <chrono>
#include <ratio>
#include <iostream>

int N = 10000;
int M = 1000;

void init( double *matrix, double *v0, double *v1, int n, int m) {
  for (int j=0; j< m; j++ ) {
    for (int i=0; i< n; i++ ) {
        matrix[i*m +j] = i + j*0.001;
    }
  }
  // v0
  for (int j=0; j< m; j++ ) {
    v0[j] = 0.1 * j;
  }
}

void initV( std::vector<double> &matrix, std::vector<double> &v0, std::vector<double> &v1, int n, int m) {
  for (int j=0; j< m; j++ ) {
    for (int i=0; i< n; i++ ) {
        matrix[i*m +j] = i + j*0.001;
    }
  }
  // v0
  for (int j=0; j< m; j++ ) {
    v0[j] = 0.1 * j;
  }
}



void matmulV( std::vector<double> &matrix, std::vector<double> &v0, std::vector<double> &v1, int n, int m) {
  for (int i=0; i< n; i++ ) {
      v1[i] = 0.0;
  }

  for (int i=0; i< n; i++ ) {
    for (int j=0; j< m; j++ ) {
      v1[i] += matrix[i*m +j] * v0[j];
    }
  }
}

void matmul( double *matrix, double *v0, double *v1, int n, int m) {
  for (int i=0; i< n; i++ ) {
      v1[i] = 0.0;
  }

  for (int i=0; i< n; i++ ) {
    for (int j=0; j< m; j++ ) {
      v1[i] += matrix[i*m +j] * v0[j];
    }
  }
}

int main( ) {

    std::vector<double> matrixV;
    matrixV.reserve(N*M);
    std::vector<double> vV0;
    vV0.reserve(M);
    std::vector<double> vV1;
    vV1.reserve(N);
    double *matrix = new double[N*M];
    double *v0 = new double[M];
    double *v1 = new double[N];
    typedef std::chrono::high_resolution_clock clock;

    auto sVector = clock::now();
    init( matrix, v0, v1, N, M);
    matmul( matrix, v0, v1, N, M);
    auto dt = clock::now() - sVector;

    auto start = clock::now();
    initV( matrixV, vV0, vV1, N, M);
    matmulV( matrixV, vV0, vV1, N, M);
    auto dtV = clock::now() - start;

    std::cout << "C array " <<  dt.count() << std::endl;
    std::cout << "vector  " <<  dtV.count() << std::endl;
    std::cout << "vector/C array  " <<  double(dtV.count())/dt.count() << std::endl;
    std::cout << "numeric test " << std::endl;

    std::cout << "  0, 0     " <<  matrix[0] << " " <<  matrixV[0] << std::endl;
    std::cout << "  0, M-1   " <<  matrix[0*M +M-1] << " " <<  matrixV[0*M +M-1] << std::endl;
    std::cout << "  N-1, 0   " <<  matrix[(N-1)*M + 0] << " " <<  matrixV[(N-1)*M + 0] << std::endl;
    std::cout << "  N-1, M-1 " <<  matrix[(N-1)*M + M-1] << " " <<  matrixV[(N-1)*M + M-1] << std::endl;


    delete [] matrix;
    delete [] v0;
    delete [] v1;
}
