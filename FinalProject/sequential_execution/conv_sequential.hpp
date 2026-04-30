#ifndef CONV_SEQUENTIAL_HPP
#define CONV_SEQUENTIAL_HPP

#include <vector>

// Forward/right-bottom padded convolution:
// output[co][h][w] = sum_ci,kh,kw input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]
// If h+kh >= H or w+kw >= W, that input contribution is treated as zero.
void conv2d_forward_sequential(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K);

#endif
