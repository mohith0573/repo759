#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "matmul.h"

int main() {
    int n = 1024;

    std::vector<double> A(n*n), B(n*n);
    std::vector<double> C1(n*n, 0.0),
                        C2(n*n, 0.0),
                        C3(n*n, 0.0),
                        C4(n*n, 0.0);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n*n; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    std::cout << n << "\n\n";

    auto run = [&](auto fn, auto& C) {
        auto start = std::chrono::high_resolution_clock::now();
        fn(A.data(), B.data(), C.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout
            << std::chrono::duration<double, std::milli>(end - start).count()
            << "\n\n"
            << C[n*n - 1] << "\n\n";
    };

    run(mmul1, C1);
    run(mmul2, C2);
    run(mmul3, C3);

    auto start = std::chrono::high_resolution_clock::now();
    mmul4(A, B, C4, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout
        << std::chrono::duration<double, std::milli>(end - start).count()
        << "\n\n"
        << C4[n*n - 1] << "\n";

    return 0;
}
