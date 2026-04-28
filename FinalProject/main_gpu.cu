#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
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

__global__ void conv2d_forward_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H, int W, int Cin, int Cout, int K);

__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H, int W, int Cin, int Cout, int K);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static void fill_deterministic(std::vector<float>& v, float scale)
{
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = scale * static_cast<float>((static_cast<int>(i) % 17) - 8);
    }
}

static void conv2d_forward_reference_cpu(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K)
{
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

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return max_diff;
}

static float time_kernel_naive(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int H, int W, int Cin, int Cout, int K,
    dim3 grid, dim3 block,
    int repeats)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up.
    conv2d_forward_naive_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        conv2d_forward_naive_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static float time_kernel_tiled(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int H, int W, int Cin, int Cout, int K,
    dim3 grid, dim3 block,
    int repeats)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up.
    conv2d_forward_tiled_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        conv2d_forward_tiled_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, Cin, Cout, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

int main(int argc, char** argv)
{
    int H = 256;
    int W = 256;
    int Cin = 8;
    int Cout = 8;
    int K = 3;
    int repeats = 20;

    if (argc >= 6) {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);
        Cin = std::atoi(argv[3]);
        Cout = std::atoi(argv[4]);
        K = std::atoi(argv[5]);
    }
    if (argc >= 7) {
        repeats = std::atoi(argv[6]);
    }

    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
        std::cerr << "Usage: ./conv_gpu H W Cin Cout K [repeats]\n";
        return 1;
    }
    if (K > MAX_K) {
        std::cerr << "Error: K=" << K << " exceeds MAX_K=" << MAX_K
                  << ". Recompile with -DMAX_K=" << K << " or use K <= " << MAX_K << ".\n";
        return 1;
    }
    if (Cout > MAX_COUT) {
        std::cerr << "Error: Cout=" << Cout << " exceeds MAX_COUT=" << MAX_COUT
                  << ". Recompile with -DMAX_COUT=" << Cout << " or use Cout <= " << MAX_COUT << ".\n";
        return 1;
    }

    const size_t input_count = static_cast<size_t>(Cin) * H * W;
    const size_t kernel_count = static_cast<size_t>(Cout) * Cin * K * K;
    const size_t output_count = static_cast<size_t>(Cout) * H * W;

    std::vector<float> h_input(input_count);
    std::vector<float> h_kernel(kernel_count);
    std::vector<float> h_ref(output_count, 0.0f);
    std::vector<float> h_naive(output_count, 0.0f);
    std::vector<float> h_tiled(output_count, 0.0f);

    fill_deterministic(h_input, 0.01f);
    fill_deterministic(h_kernel, 0.001f);

    // CPU reference is used only for correctness checking, not for GPU timing.
    conv2d_forward_reference_cpu(h_input, h_kernel, h_ref, H, W, Cin, Cout, K);

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_count * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "method,H,W,Cin,Cout,K,threads,repeats,time_ms,max_abs_diff\n";

    const float naive_ms = time_kernel_naive(d_input, d_kernel, d_output, H, W, Cin, Cout, K, grid, block, repeats);
    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    const double naive_diff = max_abs_diff(h_ref, h_naive);
    std::cout << "cuda_naive," << H << "," << W << "," << Cin << "," << Cout << "," << K
              << "," << (TILE * TILE) << "," << repeats << "," << naive_ms << "," << naive_diff << "\n";

    const float tiled_ms = time_kernel_tiled(d_input, d_kernel, d_output, H, W, Cin, Cout, K, grid, block, repeats);
    CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    const double tiled_diff = max_abs_diff(h_ref, h_tiled);
    std::cout << "cuda_tiled," << H << "," << W << "," << Cin << "," << Cout << "," << K
              << "," << (TILE * TILE) << "," << repeats << "," << tiled_ms << "," << tiled_diff << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}