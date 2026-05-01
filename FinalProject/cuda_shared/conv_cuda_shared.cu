#include "conv_cuda_shared.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define TILE 16
#define MAX_COUT 64
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err__) << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
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
    extern __shared__ float shmem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_h = blockIdx.y * blockDim.y + ty;
    const int out_w = blockIdx.x * blockDim.x + tx;

    const int tile_h = blockDim.y + K - 1;
    const int tile_w = blockDim.x + K - 1;
    const int tile_size = tile_h * tile_w;

    float acc[MAX_COUT];

    // One accumulator per output filter: this represents multiple MACs per PE.
    #pragma unroll
    for (int co = 0; co < MAX_COUT; ++co) {
        acc[co] = 0.0f;
    }

    for (int ci = 0; ci < Cin; ++ci) {
        // Cooperatively load the input tile plus right/bottom halo into shared memory.
        // This matches forward convolution: input[h+kh][w+kw].
        for (int linear = ty * blockDim.x + tx; linear < tile_size; linear += blockDim.x * blockDim.y) {
            const int local_r = linear / tile_w;
            const int local_c = linear - local_r * tile_w;

            const int global_h = blockIdx.y * blockDim.y + local_r;
            const int global_w = blockIdx.x * blockDim.x + local_c;

            float value = 0.0f;
            if (global_h < H && global_w < W) {
                value = input[(ci * H + global_h) * W + global_w];
            }
            shmem[local_r * tile_w + local_c] = value;
        }

        __syncthreads();

        if (out_h < H && out_w < W) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const float in_val = shmem[(ty + kh) * tile_w + (tx + kw)];

                    // Reuse the same input value for all output filters.
                    for (int co = 0; co < Cout; ++co) {
                        const float wgt = kernel[((co * Cin + ci) * K + kh) * K + kw];
                        acc[co] += in_val * wgt;
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

void conv2d_cuda_shared_host(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K,
    int repeats,
    float& kernel_time_ms,
    std::string& gpu_name
) {
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
        std::cerr << "ERROR: dimensions and repeats must be positive." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (Cout > MAX_COUT) {
        std::cerr << "ERROR: Cout=" << Cout << " exceeds MAX_COUT=" << MAX_COUT << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (K > TILE) {
        std::cerr << "ERROR: K=" << K << " is too large for TILE=" << TILE << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    gpu_name = prop.name;

    const size_t input_bytes = static_cast<size_t>(H) * W * Cin * sizeof(float);
    const size_t kernel_bytes = static_cast<size_t>(Cout) * Cin * K * K * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(H) * W * Cout * sizeof(float);

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    const int tile_h = TILE + K - 1;
    const int tile_w = TILE + K - 1;
    const size_t shared_bytes = static_cast<size_t>(tile_h) * tile_w * sizeof(float);

    // Warm-up launch. This prevents first-launch overhead from contaminating timing.
    conv2d_forward_shared_kernel<<<grid, block, shared_bytes>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        conv2d_forward_shared_kernel<<<grid, block, shared_bytes>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_kernel_ms, start, stop));
    kernel_time_ms = total_kernel_ms / static_cast<float>(repeats);

    CHECK_CUDA(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
}
