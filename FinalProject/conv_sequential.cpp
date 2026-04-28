#include <vector>

#ifndef MAX_COUT_CPU
#define MAX_COUT_CPU 128
#endif

// Forward/top-left anchored 2D convolution with zero padding only on bottom/right.
// Layouts:
//   input  [Cin][H][W]       index = (ci*H + h)*W + w
//   kernel [Cout][Cin][K][K] index = ((co*Cin + ci)*K + kh)*K + kw
//   output [Cout][H][W]      index = (co*H + h)*W + w
//
// Multi-MAC-per-PE model:
//   For each PE/output pixel (h,w), this function keeps one accumulator per filter.
//   The same input pixel is reused across all Cout filters before moving on.
void conv2d_forward_sequential(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K)
{
    if (Cout <= MAX_COUT_CPU) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float acc[MAX_COUT_CPU];
                for (int co = 0; co < Cout; ++co) acc[co] = 0.0f;

                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < K; ++kh) {
                        const int ih = h + kh;
                        if (ih >= H) continue; // bottom zero padding

                        for (int kw = 0; kw < K; ++kw) {
                            const int iw = w + kw;
                            if (iw >= W) continue; // right zero padding

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