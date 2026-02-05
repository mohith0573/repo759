#include <iostream>
#include <random>
#include <chrono>
#include "convolution.h"

int main(int argc, char* argv[]) {
    std::size_t n = std::stoul(argv[1]);
    std::size_t m = std::stoul(argv[2]);

    float* image = new float[n * n];
    float* mask = new float[m * m];
    float* output = new float[n * n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; i++)
        image[i] = dist_img(gen);

    for (std::size_t i = 0; i < m * m; i++)
        mask[i] = dist_mask(gen);

    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << time_ms << "\n\n";
    std::cout << output[0] << "\n\n";
    std::cout << output[n * n - 1] << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
