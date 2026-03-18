#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include "matmul.h"

int main(int argc, char* argv[]) {

    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);

    omp_set_num_threads(t);

    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n]();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < n*n; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    double time =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << C[0] << "\n";
    std::cout << C[n*n - 1] << "\n";
    std::cout << time << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
