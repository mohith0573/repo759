#include "conv_cuda_naive.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                                 \
    } while (0)

// Keep this larger than the Cout values you plan to test.
// Your current project uses Cout=8, so this is safely above that.
constexpr int MAX_COUT = 64;
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

__global__ void conv2d_forward_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K
) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (h >= H || w >= W) {
        return;
    }

    // One CUDA thread = one PE = one output pixel location.
    // Multiple MACs per PE are modeled by one accumulator per output filter.
    float acc[MAX_COUT];

    for (int co = 0; co < Cout; ++co) {
        acc[co] = 0.0f;
    }

    // Forward/right-bottom convolution:
    // output[h][w] uses input[h + kh][w + kw].
    // Bottom/right out-of-bound accesses are treated as zero padding.
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            const int ih = h + kh;
            if (ih >= H) {
                continue;
            }

            for (int kw = 0; kw < K; ++kw) {
                const int iw = w + kw;
                if (iw >= W) {
                    continue;
                }

                const float in_val = input[(ci * H + ih) * W + iw];

                // Reuse the same input value across all Cout filters.
                // This is more locality-friendly than recomputing the input window separately per filter.
                for (int co = 0; co < Cout; ++co) {
                    const int kidx = ((co * Cin + ci) * K + kh) * K + kw;
                    acc[co] += in_val * kernel[kidx];
                }
            }
        }
    }

    for (int co = 0; co < Cout; ++co) {
        output[(co * H + h) * W + w] = acc[co];
    }
}

void run_cuda_naive_convolution(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H,
    int W,
    int Cin,
    int Cout,
    int K,
    int repeats,
    float& avg_kernel_time_ms,
    float& total_time_ms,
    std::string& gpu_name
) {
    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
        throw std::runtime_error("Invalid dimensions or repeats.");
    }

    if (Cout > MAX_COUT) {
        throw std::runtime_error("Cout exceeds MAX_COUT. Increase MAX_COUT in conv_cuda_naive.cu.");
    }

    const size_t input_count = static_cast<size_t>(H) * W * Cin;
    const size_t kernel_count = static_cast<size_t>(Cout) * Cin * K * K;
    const size_t output_count = static_cast<size_t>(H) * W * Cout;

    if (input.size() != input_count) {
        throw std::runtime_error("Input vector size does not match H*W*Cin.");
    }
    if (kernel.size() != kernel_count) {
        throw std::runtime_error("Kernel vector size does not match Cout*Cin*K*K.");
    }
    if (output.size() != output_count) {
        output.assign(output_count, 0.0f);
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    gpu_name = prop.name;

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    const size_t input_bytes = input_count * sizeof(float);
    const size_t kernel_bytes = kernel_count * sizeof(float);
    const size_t output_bytes = output_count * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);

    auto total_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

    // Warm-up launch: avoids including first-launch overhead in timed kernel loop.
    conv2d_forward_naive_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event));
    for (int r = 0; r < repeats; ++r) {
        conv2d_forward_naive_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float repeated_kernel_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&repeated_kernel_time_ms, start_event, stop_event));
    avg_kernel_time_ms = repeated_kernel_time_ms / static_cast<float>(repeats);

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto total_end = std::chrono::high_resolution_clock::now();
    total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}
