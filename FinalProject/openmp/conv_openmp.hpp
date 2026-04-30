#ifndef CONV_OPENMP_HPP
#define CONV_OPENMP_HPP

#include <vector>

// Forward/right-bottom padded convolution.
// input layout:  input[ci][h][w]       -> (ci * H + h) * W + w
// kernel layout: kernel[co][ci][kh][kw] -> ((co * Cin + ci) * K + kh) * K + kw
// output layout: output[co][h][w]      -> (co * H + h) * W + w
void conv2d_forward_openmp(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K
);

#endif
