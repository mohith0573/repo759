#include "conv_sequential.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

static inline int input_index(int ci, int h, int w, int H, int W) {
    return (ci * H + h) * W + w;
}

static inline int output_index(int co, int h, int w, int H, int W) {
    return (co * H + h) * W + w;
}

static inline int kernel_index(int co, int ci, int kh, int kw, int Cin, int K) {
    return ((co * Cin + ci) * K + kh) * K + kw;
}

void conv2d_forward_sequential(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K)
{
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0) {
        throw std::invalid_argument("H, W, Cin, Cout, and K must all be positive.");
    }

    const std::size_t expected_input  = static_cast<std::size_t>(H) * W * Cin;
    const std::size_t expected_kernel = static_cast<std::size_t>(Cout) * Cin * K * K;
    const std::size_t expected_output = static_cast<std::size_t>(Cout) * H * W;

    if (input.size() != expected_input) {
        throw std::invalid_argument("input size does not match H*W*Cin.");
    }
    if (kernel.size() != expected_kernel) {
        throw std::invalid_argument("kernel size does not match Cout*Cin*K*K.");
    }

    output.assign(expected_output, 0.0f);

    // Reorder kernel once to improve locality in the inner Cout loop.
    // Original: kernel[co][ci][kh][kw]
    // Reordered: kernel_reordered[ci][kh][kw][co]
    std::vector<float> kernel_reordered(expected_kernel);
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int base = ((ci * K + kh) * K + kw) * Cout;
                for (int co = 0; co < Cout; ++co) {
                    kernel_reordered[base + co] = kernel[kernel_index(co, ci, kh, kw, Cin, K)];
                }
            }
        }
    }

    std::vector<float> acc(Cout, 0.0f);

    // PE mapping: one PE per output pixel (h,w).
    // Multi-MAC-per-PE model: acc[co] represents one MAC accumulator per output filter.
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            std::fill(acc.begin(), acc.end(), 0.0f);

            for (int ci = 0; ci < Cin; ++ci) {
                for (int kh = 0; kh < K; ++kh) {
                    const int ih = h + kh;
                    if (ih >= H) {
                        continue;  // bottom padding
                    }

                    for (int kw = 0; kw < K; ++kw) {
                        const int iw = w + kw;
                        if (iw >= W) {
                            continue;  // right padding
                        }

                        const float in_val = input[input_index(ci, ih, iw, H, W)];
                        const int kbase = ((ci * K + kh) * K + kw) * Cout;

                        // Reuse one input value across all output filters.
                        for (int co = 0; co < Cout; ++co) {
                            acc[co] += in_val * kernel_reordered[kbase + co];
                        }
                    }
                }
            }

            for (int co = 0; co < Cout; ++co) {
                output[output_index(co, h, w, H, W)] = acc[co];
            }
        }
    }
}
