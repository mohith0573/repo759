#include "matmul.cuh"
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * n;

    if (idx >= total) return;

    size_t row = idx / n;
    size_t col = idx % n;

    float sum = 0.0f;
    for (size_t k = 0; k < n; ++k)
        sum += A[row * n + k] * B[k * n + col];

    C[idx] = sum;
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    size_t total = n * n;
    size_t blocks = (total + threads_per_block - 1) / threads_per_block;

    matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
}
