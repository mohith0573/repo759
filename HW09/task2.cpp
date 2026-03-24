#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "montecarlo.h"

int main(int argc, char* argv[]) {

    size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);

    std::vector<float> x(n), y(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }

    omp_set_num_threads(t);

    auto start = std::chrono::high_resolution_clock::now();

    int incircle = montecarlo(n, x.data(), y.data(), 1.0f);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    double pi = 4.0 * incircle / n;

    std::cout << pi << "\n";
    std::cout << time_ms << "\n";

    return 0;
}
