#include <iostream>
#include <random>
#include <chrono>
#include <cstddef>
#include "scan.h"

int main(int argc, char* argv[]) {
    std::size_t n = std::stoul(argv[1]);

    float* input = new float[n];
    float* output = new float[n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n; i++) {
        input[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    scan(input, output, n);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << time_ms << "\n\n";
    std::cout << output[0] << "\n\n";
    std::cout << output[n - 1] << "\n";

    delete[] input;
    delete[] output;

    return 0;
}
