#include "conv_openmp.hpp"
#include <algorithm>
#include <omp.h>

// Forward / top-left anchored convolution with bottom/right zero padding.
//
// input  layout: input[ci][h][w]
// kernel layout: kernel[co][ci][kh][kw]
// output layout: output[co][h][w]
//
// For each output PE position (h,w), the PE computes all Cout filters.
// A private accumulator vector is allocated once per OpenMP thread and reused
// for every PE assigned to that thread. This avoids races and improves locality.
void conv2d_forward_openmp(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K,
    int num_threads
) {
    const int HW = H * W;

    // One parallel region avoids repeatedly creating/destroying threads inside loops.
    #pragma omp parallel num_threads(num_threads)
    {
        // Private per-thread accumulators: no race condition.
        std::vector<float> acc(static_cast<size_t>(Cout), 0.0f);

        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {

                std::fill(acc.begin(), acc.end(), 0.0f);

                for (int ci = 0; ci < Cin; ++ci) {
                    const int input_channel_base = ci * HW;

                    for (int kh = 0; kh < K; ++kh) {
                        const int ih = h + kh;
                        if (ih >= H) continue; // bottom zero padding

                        for (int kw = 0; kw < K; ++kw) {
                            const int iw = w + kw;
                            if (iw >= W) continue; // right zero padding

                            const float in_val = input[input_channel_base + ih * W + iw];

                            // Reuse the same input value across all output filters.
                            for (int co = 0; co < Cout; ++co) {
                                const int kidx = ((co * Cin + ci) * K + kh) * K + kw;
                                acc[co] += in_val * kernel[kidx];
                            }
                        }
                    }
                }

                const int out_pixel_offset = h * W + w;
                for (int co = 0; co < Cout; ++co) {
                    output[co * HW + out_pixel_offset] = acc[co];
                }
            }
        }
    }
}
