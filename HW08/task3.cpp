#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include "msort.h"

int main(int argc, char* argv[]) {
    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);
    std::size_t ts = std::stoul(argv[3]);

    omp_set_num_threads(t);

    int* arr = new int[n];

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (std::size_t i = 0; i < n; i++)
        arr[i] = dist(gen);

    auto start = std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << arr[0] << "\n";
    std::cout << arr[n - 1] << "\n";
    std::cout << time_ms << "\n";

    delete[] arr;
    return 0;
}
