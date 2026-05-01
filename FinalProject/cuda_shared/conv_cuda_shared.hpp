#ifndef CONV_CUDA_SHARED_HPP
#define CONV_CUDA_SHARED_HPP

#include <cstddef>

// Host wrapper for CUDA shared-memory tiled forward convolution.
// Layouts:
//   input  [Cin][H][W]        index = (ci * H + h) * W + w
//   kernel [Cout][Cin][K][K]  index = ((co * Cin + ci) * K + kh) * K + kw
//   output [Cout][H][W]       index = (co * H + h) * W + w
// Convolution rule:
//   output[co][h][w] = sum input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]
// Boundary: bottom/right zero padding when h+kh >= H or w+kw >= W.

float run_cuda_shared_convolution(
    const float* h_input,
    const float* h_kernel,
    float* h_output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K,
    int repeats,
    int* threads_per_block_out,
    char* gpu_name,
    std::size_t gpu_name_capacity
);

#endif
