#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

#ifndef MAX_K
#define MAX_K 7
#endif

#ifndef MAX_COUT
#define MAX_COUT 64
#endif

// Shared-memory tiled CUDA kernel for forward/top-left anchored convolution.
// One thread = one PE/output pixel position (h,w).
// A block computes TILE x TILE output pixels.
// For each input channel, the block cooperatively loads a (TILE+K-1) x (TILE+K-1)
// input tile into shared memory. The extra K-1 rows/columns are the bottom/right halo.
// Then each PE/thread reuses that tile and updates all output-filter accumulators.
__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H, int W, int Cin, int Cout, int K)
{
    __shared__ float tile[TILE + MAX_K - 1][TILE + MAX_K - 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int h = blockIdx.y * TILE + ty;
    const int w = blockIdx.x * TILE + tx;

    float acc[MAX_COUT];
    for (int co = 0; co < Cout; ++co) acc[co] = 0.0f;

    for (int ci = 0; ci < Cin; ++ci) {
        // Cooperative load of input channel tile + bottom/right halo.
        // For K=3 and TILE=16, this loads 18x18 values for 16x16 PEs.
        for (int local_y = ty; local_y < TILE + K - 1; local_y += blockDim.y) {
            const int ih = blockIdx.y * TILE + local_y;

            for (int local_x = tx; local_x < TILE + K - 1; local_x += blockDim.x) {
                const int iw = blockIdx.x * TILE + local_x;

                if (ih < H && iw < W) {
                    tile[local_y][local_x] = input[(ci * H + ih) * W + iw];
                } else {
                    tile[local_y][local_x] = 0.0f; // bottom/right zero padding
                }
            }
        }

        __syncthreads();

        if (h < H && w < W) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const float x = tile[ty + kh][tx + kw];

                    for (int co = 0; co < Cout; ++co) {
                        const float wt = kernel[((co * Cin + ci) * K + kh) * K + kw];
                        acc[co] += x * wt;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (h < H && w < W) {
        for (int co = 0; co < Cout; ++co) {
            output[(co * H + h) * W + w] = acc[co];
        }
    }
}