#include "conv_openmp.hpp"

#include <algorithm>
#include <omp.h>
#include <vector>

void conv2d_forward_openmp(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K
) {
    // Locality-oriented OpenMP implementation:
    // One OpenMP iteration computes one PE/output pixel (h,w).
    // For that PE, keep Cout accumulators locally and reuse each input value
    // across all output filters before loading the next input value.

#pragma omp parallel
    {
        std::vector<float> acc(static_cast<size_t>(Cout), 0.0f);

#pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                std::fill(acc.begin(), acc.end(), 0.0f);

                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < K; ++kh) {
                        const int ih = h + kh;
                        if (ih >= H) {
                            continue; // bottom zero padding
                        }

                        for (int kw = 0; kw < K; ++kw) {
                            const int iw = w + kw;
                            if (iw >= W) {
                                continue; // right zero padding
                            }

                            const float in_val = input[(static_cast<size_t>(ci) * H + ih) * W + iw];

                            // Reuse the same input value across all filters/MACs.
                            for (int co = 0; co < Cout; ++co) {
                                const size_t kidx = ((static_cast<size_t>(co) * Cin + ci) * K + kh) * K + kw;
                                acc[co] += in_val * kernel[kidx];
                            }
                        }
                    }
                }

                for (int co = 0; co < Cout; ++co) {
                    output[(static_cast<size_t>(co) * H + h) * W + w] = acc[co];
                }
            }
        }
    }
}
