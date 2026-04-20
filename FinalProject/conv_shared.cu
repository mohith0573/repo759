// conv_shared.cu  ── CUDA shared-memory tiled 2D convolution
//
// Usage:  ./conv_shared  H  W  C_in  C_out  K
//
// Thread mapping:
//   One CUDA thread ↔ one output pixel (h, w)  ↔ one PE
//   Each thread runs C_out MACs; each MAC loops over (ci, kh, kw).
//
// Shared memory strategy (mirrors paper's on-chip register file):
//   For each input channel ci, threads in a block cooperatively load a
//   spatial tile — including the (K/2)-pixel halo needed by border threads —
//   into __shared__ memory.  All C_out MACs for every PE then read from
//   shared memory instead of global memory.
//
//   Shared mem size per block: (TILE + K - 1)² × sizeof(float)
//     e.g. TILE=16, K=3 → 18×18×4 = 1296 bytes  (very small)
//
//   This directly mirrors the paper's concept:
//     same input pixel reused by all C_out MACs of the same PE,
//     and by neighbouring PEs in the same tile — without re-fetching
//     from external (global) memory.
//
// Output:  CSV line → gpu_shared,H,W,C_in,C_out,K,1,<time_ms>
//          Binary   → results/out_shared.bin
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "common.h"

#define TILE 16   // spatial tile edge; thread block = TILE × TILE

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ── Kernel ───────────────────────────────────────────────────────────────────
__global__ void kernel_shared(const float* __restrict__ input,
                               const float* __restrict__ weight,
                               float*       __restrict__ output,
                               int H, int W, int C_in, int C_out, int K)
{
    // Shared memory tile: (TILE + K - 1)² elements for one input channel
    extern __shared__ float smem[];

    int tx = threadIdx.x, ty = threadIdx.y;

    // Global output pixel coordinates owned by this thread (= this PE)
    int w = blockIdx.x * TILE + tx;
    int h = blockIdx.y * TILE + ty;

    const int pad      = K / 2;
    const int smem_w   = TILE + K - 1;   // shared tile width  (includes halo)
    const int smem_h   = TILE + K - 1;   // shared tile height (includes halo)
    const int smem_tot = smem_w * smem_h;
    const int tid      = ty * TILE + tx;      // flat thread ID within block
    const int nthreads = TILE * TILE;

    // ── Initialize output accumulator in global memory to 0 ─────────────
    // We accumulate across ci iterations directly into global output.
    // Each thread exclusively owns its (h, w) output slots → no races.
    if (h < H && w < W) {
        for (int co = 0; co < C_out; co++)
            output[h*W*C_out + w*C_out + co] = 0.0f;
    }

    // ── Outer loop: one input channel at a time ──────────────────────────
    // Load the spatial tile for ci into shared memory, then all C_out MACs
    // of every PE in the block use it — exactly like the paper's on-chip RF.
    for (int ci = 0; ci < C_in; ci++) {

        // Cooperative tile load — threads share the work round-robin
        for (int idx = tid; idx < smem_tot; idx += nthreads) {
            int sm_row = idx / smem_w;
            int sm_col = idx % smem_w;
            // Global coords of this shared-memory element
            int ih = blockIdx.y * TILE + sm_row - pad;
            int iw = blockIdx.x * TILE + sm_col - pad;
            smem[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        ? input[ih*W*C_in + iw*C_in + ci]
                        : 0.0f;
        }
        __syncthreads();   // all threads have finished loading the tile

        // ── Each active PE runs all C_out MACs using shared memory ───────
        if (h < H && w < W) {
            for (int co = 0; co < C_out; co++) {   // C_out MACs per PE

                float acc = 0.0f;   // partial sum for this channel slice

                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        // smem index: thread's local position + kernel offset
                        float in_val  = smem[(ty + kh) * smem_w + (tx + kw)];
                        float wt_val  = weight[co*K*K*C_in + kh*K*C_in + kw*C_in + ci];
                        acc += in_val * wt_val;
                    }
                }
                // Accumulate partial channel sum into global output
                output[h*W*C_out + w*C_out + co] += acc;
            }
        }
        __syncthreads();   // safe to overwrite smem in the next ci iteration
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

    float* h_input  = new float[in_n];
    float* h_weight = new float[wt_n];
    float* h_output = new float[out_n];

    fill_random(h_input,  in_n,  42u);
    fill_random(h_weight, wt_n, 123u);

    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  in_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, wt_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, out_n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input,  h_input,  in_n  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, wt_n  * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    // Shared memory size: one spatial tile (with halo) per block
    int smem_bytes = (TILE + K - 1) * (TILE + K - 1) * sizeof(float);

    // Warm-up
    kernel_shared<<<grid, block, smem_bytes>>>(d_input, d_weight, d_output, H, W, C_in, C_out, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel_shared<<<grid, block, smem_bytes>>>(d_input, d_weight, d_output, H, W, C_in, C_out, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    save_output("results/out_shared.bin", h_output, out_n);

    printf("gpu_shared,%d,%d,%d,%d,%d,1,%.4f\n", H, W, C_in, C_out, K, elapsed_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
    delete[] h_input; delete[] h_weight; delete[] h_output;
    return 0;
}
