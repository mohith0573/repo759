// conv_omp.cpp  ── OpenMP multi-core 2D convolution
//
// Usage:  ./conv_omp  H  W  C_in  C_out  K  T
//           T = number of OpenMP threads
//
// Parallelism strategy:
//   The outer (h, w) loops are collapsed and distributed across T threads.
//   Each thread independently handles a subset of output pixels (PEs).
//   Within each PE, C_out MACs run sequentially (same as the sequential impl).
//   No shared mutable state → no race conditions.
//
// Output:  CSV line → omp,H,W,C_in,C_out,K,T,<time_ms>
//          Binary   → results/out_omp.bin
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "common.h"

// ── Core kernel ──────────────────────────────────────────────────────────────
void conv_openmp(const float* __restrict__ input,
                 const float* __restrict__ weight,
                 float*       __restrict__ output,
                 int H, int W, int C_in, int C_out, int K,
                 int num_threads)
{
    const int pad = K / 2;

    omp_set_num_threads(num_threads);

    // Collapse h and w into one loop → better load balancing across threads
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {

            // ── This pixel is one PE ─────────────────────────────────────
            // Each thread owns its private (h, w) — no shared writes.
            for (int co = 0; co < C_out; co++) {   // C_out MACs per PE

                float acc = 0.0f;

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
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s H W C_in C_out K T\n", argv[0]);
        return 1;
    }
    int H    = atoi(argv[1]), W    = atoi(argv[2]);
    int C_in = atoi(argv[3]), C_out = atoi(argv[4]);
    int K    = atoi(argv[5]), T     = atoi(argv[6]);

    long in_n  = input_size (H, W, C_in);
    long wt_n  = weight_size(C_out, K, C_in);
    long out_n = output_size(H, W, C_out);

    float* input  = new float[in_n];
    float* weight = new float[wt_n];
    float* output = new float[out_n];

    fill_random(input,  in_n,  42u);
    fill_random(weight, wt_n, 123u);

    double t0 = now_ms();
    conv_openmp(input, weight, output, H, W, C_in, C_out, K, T);
    double elapsed = now_ms() - t0;

    save_output("results/out_omp.bin", output, out_n);
    printf("omp,%d,%d,%d,%d,%d,%d,%.4f\n", H, W, C_in, C_out, K, T, elapsed);

    delete[] input;
    delete[] weight;
    delete[] output;
    return 0;
}
