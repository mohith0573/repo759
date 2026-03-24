#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "convolve.h"

int main(int argc, char* argv[]) {

    size_t n = std::stoul(argv[1]);

    std::vector<float> image(n*n);
    std::vector<float> output(n*n);
    std::vector<float> mask(9);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto &v : image) v = dist(gen);
    for (auto &v : mask) v = dist(gen);

    auto start = std::chrono::high_resolution_clock::now();

    convolve(image.data(), output.data(), n, mask.data(), 3);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << time_ms << std::endl;

    return 0;
}
