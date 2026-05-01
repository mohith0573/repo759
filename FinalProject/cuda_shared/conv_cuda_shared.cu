#include "conv_cuda_shared.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int MAX_COUT = 64;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
    } while (0)

__global__ void conv2d_forward_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K
) {
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int threads_per_block = blockDim.x * blockDim.y;

    const int out_h = blockIdx.y * blockDim.y + ty;
    const int out_w = blockIdx.x * blockDim.x + tx;

    const int tile_w = blockDim.x + K - 1;
    const int tile_h = blockDim.y + K - 1;
    const int tile_size = tile_w * tile_h;

    float acc[MAX_COUT];

    for (int co = 0; co < MAX_COUT; ++co) {
        acc[co] = 0.0f;
    }

    // One shared-memory tile is loaded per input channel.
    // The tile covers the block output region plus the right/bottom halo.
    for (int ci = 0; ci < Cin; ++ci) {
        for (int idx = tid; idx < tile_size; idx += threads_per_block) {
            const int local_y = idx / tile_w;
            const int local_x = idx - local_y * tile_w;

            const int in_h = blockIdx.y * blockDim.y + local_y;
            const int in_w = blockIdx.x * blockDim.x + local_x;

            if (in_h < H && in_w < W) {
                tile[idx] = input[(ci * H + in_h) * W + in_w];
            } else {
                tile[idx] = 0.0f;
            }
        }

        __syncthreads();

        if (out_h < H && out_w < W) {
            // Multi-MAC per PE: one independent accumulator per output filter.
            // The input tile value is reused across all filters.
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const float in_val = tile[(ty + kh) * tile_w + (tx + kw)];

                    for (int co = 0; co < Cout; ++co) {
                        const float w_val = kernel[((co * Cin + ci) * K + kh) * K + kw];
                        acc[co] += in_val * w_val;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (out_h < H && out_w < W) {
        for (int co = 0; co < Cout; ++co) {
            output[(co * H + out_h) * W + out_w] = acc[co];
        }
    }
}

}  // namespace

float run_cuda_shared_convolution(
    const float* h_input,
    const float* h_kernel,
    float* h_output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K,
    int repeats,
    int* threads_per_block_out,
    char* gpu_name,
    std::size_t gpu_name_capacity
) {
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
        throw std::runtime_error("Invalid convolution dimensions or repeats.");
    }
    if (Cout > MAX_COUT) {
        throw std::runtime_error("Cout exceeds MAX_COUT=64 for this CUDA shared implementation.");
    }
    if (K > 15) {
        throw std::runtime_error("K is very large for this tiled implementation. Use K <= 15.");
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (gpu_name && gpu_name_capacity > 0) {
        std::snprintf(gpu_name, gpu_name_capacity, "%s", prop.name);
    }

    const std::size_t input_count = static_cast<std::size_t>(H) * W * Cin;
    const std::size_t kernel_count = static_cast<std::size_t>(Cout) * Cin * K * K;
    const std::size_t output_count = static_cast<std::size_t>(H) * W * Cout;

    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t kernel_bytes = kernel_count * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    if (threads_per_block_out) {
        *threads_per_block_out = static_cast<int>(block.x * block.y);
    }

    const int tile_w = static_cast<int>(block.x) + K - 1;
    const int tile_h = static_cast<int>(block.y) + K - 1;
    const std::size_t shared_bytes = static_cast<std::size_t>(tile_w) * tile_h * sizeof(float);

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch to reduce one-time overhead from the timed section.
    conv2d_forward_shared_kernel<<<grid, block, shared_bytes>>>(
        d_input, d_kernel, d_output, H, W, Cin, Cout, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        conv2d_forward_shared_kernel<<<grid, block, shared_bytes>>>(
            d_input, d_kernel, d_output, H, W, Cin, Cout, K
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, start, stop));
    const float avg_kernel_ms = total_kernel_ms / static_cast<float>(repeats);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    return avg_kernel_ms;
}
