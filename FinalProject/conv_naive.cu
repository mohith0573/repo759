// conv_naive.cu  ── CUDA naive (global memory only) 2D convolution
//
// Usage:  ./conv_naive  H  W  C_in  C_out  K
//
// Thread mapping:
//   One CUDA thread ↔ one output pixel (h, w)  ↔ one PE
//   Each thread runs C_out MACs back-to-back; each MAC loops over
//   all (ci, kh, kw) reading directly from global memory.
//
// This is the baseline GPU implementation — no data reuse.
// Every MAC re-fetches the same input pixels from global memory.
//
// Output:  CSV line → gpu_naive,H,W,C_in,C_out,K,1,<time_ms>
//          Binary   → results/out_naive.bin
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "common.h"

#define TILE 16   // 16×16 thread block = 256 threads per block

// ── CUDA error check ─────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ── Kernel ───────────────────────────────────────────────────────────────────
// Each thread computes ALL output channels for its pixel.
// Reads input from global memory for every MAC — no caching.
__global__ void kernel_naive(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              float*       __restrict__ output,
                              int H, int W, int C_in, int C_out, int K)
{
    int w = blockIdx.x * TILE + threadIdx.x;   // output column
    int h = blockIdx.y * TILE + threadIdx.y;   // output row

    if (h >= H || w >= W) return;

    const int pad = K / 2;

    // ── This thread is one PE ────────────────────────────────────────────
    // Loop over C_out MACs; each produces one output channel value.
    for (int co = 0; co < C_out; co++) {

        float acc = 0.0f;   // accumulator for MAC[co]

        // Accumulate over all input channels and kernel window
        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    int ih = h + kh - pad;
                    int iw = w + kw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        acc += input [ih*W*C_in   + iw*C_in  + ci]
                             * weight[co*K*K*C_in + kh*K*C_in + kw*C_in + ci];
                    }
                }
            }
        }
        output[h*W*C_out + w*C_out + co] = acc;
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s H W C_in C_out K\n", argv[0]);
        return 1;
    }
    int H = atoi(argv[1]), W = atoi(argv[2]);
    int C_in = atoi(argv[3]), C_out = atoi(argv[4]), K = atoi(argv[5]);

    long in_n  = input_size (H, W, C_in);
    long wt_n  = weight_size(C_out, K, C_in);
    long out_n = output_size(H, W, C_out);

    // Host allocation + data generation (same seed as other impls)
    float* h_input  = new float[in_n];
    float* h_weight = new float[wt_n];
    float* h_output = new float[out_n];

    fill_random(h_input,  in_n,  42u);
    fill_random(h_weight, wt_n, 123u);

    // Device allocation
    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  in_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, wt_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, out_n * sizeof(float)));

    // H→D transfer
    CUDA_CHECK(cudaMemcpy(d_input,  h_input,  in_n  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, wt_n  * sizeof(float), cudaMemcpyHostToDevice));

    // Grid / block dims: one thread per output pixel
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    // Warm-up
    kernel_naive<<<grid, block>>>(d_input, d_weight, d_output, H, W, C_in, C_out, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run — use CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel_naive<<<grid, block>>>(d_input, d_weight, d_output, H, W, C_in, C_out, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // D→H + save
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    save_output("results/out_naive.bin", h_output, out_n);

    printf("gpu_naive,%d,%d,%d,%d,%d,1,%.4f\n", H, W, C_in, C_out, K, elapsed_ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
    delete[] h_input; delete[] h_weight; delete[] h_output;
    return 0;
}
