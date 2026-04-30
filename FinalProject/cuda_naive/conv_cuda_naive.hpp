#ifndef CONV_CUDA_NAIVE_HPP
#define CONV_CUDA_NAIVE_HPP

#include <string>
#include <vector>

// Host-side launcher for the CUDA naive forward/right-bottom convolution.
// Input layout : input[ci][h][w]       -> (ci * H + h) * W + w
// Kernel layout: kernel[co][ci][kh][kw]-> ((co * Cin + ci) * K + kh) * K + kw
// Output layout: output[co][h][w]      -> (co * H + h) * W + w
void run_cuda_naive_convolution(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K,
    int repeats,
    float& avg_kernel_time_ms,
    float& total_time_ms,
    std::string& gpu_name
);

#endif
