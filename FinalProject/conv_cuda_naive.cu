#include <cuda_runtime.h>

#ifndef MAX_COUT
#define MAX_COUT 64
#endif

// Naive CUDA kernel: one thread = one PE/output pixel position (h,w).
// Each thread has multiple conceptual MACs: one accumulator per output filter.
// The same input value x is reused across all Cout filters before moving on.
// All input and weight reads come directly from global memory.
// Layouts:
//   input  [Cin][H][W]
//   kernel [Cout][Cin][K][K]
//   output [Cout][H][W]
__global__ void conv2d_forward_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H, int W, int Cin, int Cout, int K)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (h >= H || w >= W) return;

    float acc[MAX_COUT];
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