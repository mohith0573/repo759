#ifndef CONV_CUDA_SHARED_HPP
#define CONV_CUDA_SHARED_HPP

#include <string>
#include <vector>

// CUDA shared-memory tiled convolution.
// Input layout : input[ci][h][w]          -> (ci * H + h) * W + w
// Kernel layout: kernel[co][ci][kh][kw]   -> ((co * Cin + ci) * K + kh) * K + kw
// Output layout: output[co][h][w]         -> (co * H + h) * W + w
// Convolution rule: output[co][h][w] += input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]
// Boundary: bottom/right zero padding when h+kh >= H or w+kw >= W.
void conv2d_cuda_shared_host(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K,
    int repeats,
    float& kernel_time_ms,
    std::string& gpu_name
);

#endif
