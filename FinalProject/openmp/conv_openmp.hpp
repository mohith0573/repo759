#ifndef CONV_OPENMP_HPP
#define CONV_OPENMP_HPP

#include <vector>

void conv2d_forward_openmp(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K,
    int num_threads
);

#endif
