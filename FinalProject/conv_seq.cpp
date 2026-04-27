// conv_seq.cpp  ── Sequential (single-threaded) 2D convolution
//
// Usage:  ./conv_seq  H  W  C_in  C_out  K
//
// Each output pixel (h, w) acts as one PE.
// That PE runs C_out MACs in sequence; MAC[co] accumulates over all
// input channels (ci) and kernel positions (kh, kw).
//
// Output:  CSV line → seq,H,W,C_in,C_out,K,1,<time_ms>
//          Binary   → results/out_seq.bin
// 

#include <cstdio>
#include <cstdlib>
#include "common.h"

// ── Core kernel ──────────────────────────────────────────────────────────────
void conv_sequential(const float* __restrict__ input,
                     const float* __restrict__ weight,
                     float*       __restrict__ output,
                     int H, int W, int C_in, int C_out, int K)
{
    const int pad = K / 2;

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {

            // ── This pixel is one PE ─────────────────────────────────────
            // C_out MACs run for this PE; each MAC owns one output channel.
            for (int co = 0; co < C_out; co++) {

                float acc = 0.0f;   // MAC accumulator for filter 'co'

                // Accumulate over all input channels and kernel positions
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = h + kh - pad;
                            int iw = w + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                acc += input [ih*W*C_in  + iw*C_in  + ci]
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
    if (argc != 6) {
        fprintf(stderr, "Usage: %s H W C_in C_out K\n", argv[0]);
        return 1;
    }
    int H = atoi(argv[1]), W = atoi(argv[2]);
    int C_in = atoi(argv[3]), C_out = atoi(argv[4]), K = atoi(argv[5]);

    // Allocate
    long in_n  = input_size (H, W, C_in);
    long wt_n  = weight_size(C_out, K, C_in);
    long out_n = output_size(H, W, C_out);

    float* input  = new float[in_n];
    float* weight = new float[wt_n];
    float* output = new float[out_n];

    // Same seed across all implementations → identical data
    fill_random(input,  in_n,  42u);
    fill_random(weight, wt_n, 123u);

    // Timed run
    double t0 = now_ms();
    conv_sequential(input, weight, output, H, W, C_in, C_out, K);
    double elapsed = now_ms() - t0;

    // Results
    save_output("results/out_seq.bin", output, out_n);
    printf("seq,%d,%d,%d,%d,%d,1,%.4f\n", H, W, C_in, C_out, K, elapsed);

    delete[] input;
    delete[] weight;
    delete[] output;
    return 0;
}
