#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "cluster.h"

int main(int argc, char* argv[]) {
    size_t n = std::stoul(argv[1]);
    size_t t = std::stoul(argv[2]);

    std::vector<float> arr(n);
    std::vector<float> centers(t);
    std::vector<float> dists(t, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, (float)n);

    for (size_t i = 0; i < n; i++)
        arr[i] = dist(gen);

    std::sort(arr.begin(), arr.end());

    for (size_t i = 0; i < t; i++) {
        centers[i] = ((2*i + 1) * n) / (2.0f * t);
    }

    auto start = std::chrono::high_resolution_clock::now();

    cluster(n, t, arr.data(), centers.data(), dists.data());

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    float max_val = dists[0];
    int max_id = 0;

    for (size_t i = 1; i < t; i++) {
        if (dists[i] > max_val) {
            max_val = dists[i];
            max_id = i;
        }
    }

    std::cout << max_val << "\n";
    std::cout << max_id << "\n";
    std::cout << time_ms << "\n";

    return 0;
}
