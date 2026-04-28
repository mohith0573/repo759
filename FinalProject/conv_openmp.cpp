#include <vector>
#include <omp.h>

#ifndef MAX_COUT_CPU
#define MAX_COUT_CPU 128
#endif

// Forward/top-left anchored 2D convolution with zero padding only on bottom/right.
// Same layout as conv_sequential.cpp.
// Parallelization: independent PE/output-pixel locations are distributed across OpenMP threads.
void conv2d_forward_openmp(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K)
{
    if (Cout <= MAX_COUT_CPU) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float acc[MAX_COUT_CPU];
                for (int co = 0; co < Cout; ++co) acc[co] = 0.0f;

                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < K; ++kh) {
                        const int ih = h + kh;
                        if (ih >= H) continue;

                        for (int kw = 0; kw < K; ++kw) {
                            const int iw = w + kw;
                            if (iw >= W) continue;

                            const float x = input[(ci * H + ih) * W + iw];

                            for (int co = 0; co < Cout; ++co) {
                                const float wt = kernel[((co * Cin + ci) * K + kh) * K + kw];
                                acc[co] += x * wt;
                            }
                        }
                    }
                }

                for (int co = 0; co < Cout; ++co) {
                    output[(co * H + h) * W + w] = acc[co];
                }
            }
        }
        return;
    }

    // Robust fallback for Cout > MAX_COUT_CPU.
    #pragma omp parallel for collapse(3) schedule(static)
    for (int co = 0; co < Cout; ++co) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float sum = 0.0f;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < K; ++kh) {
                        const int ih = h + kh;
                        if (ih >= H) continue;
                        for (int kw = 0; kw < K; ++kw) {
                            const int iw = w + kw;
                            if (iw >= W) continue;
                            const float x = input[(ci * H + ih) * W + iw];
                            const float wt = kernel[((co * Cin + ci) * K + kh) * K + kw];
                            sum += x * wt;
                        }
                    }
                }
                output[(co * H + h) * W + w] = sum;
            }
        }
    }
}