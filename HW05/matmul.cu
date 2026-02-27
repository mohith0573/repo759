#include <cuda_runtime.h>
#include "matmul.cuh"
#include <algorithm> // for min

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n)
{
    extern __shared__ T smem[];

    T *As = smem;
    T *Bs = &smem[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    T sum = 0;

    for (int t = 0; t < (n + blockDim.x - 1) / blockDim.x; t++)
    {
        int tiledCol = t * blockDim.x + tx;
        int tiledRow = t * blockDim.y + ty;

        As[ty * blockDim.x + tx] = (row < n && tiledCol < n) ? A[row * n + tiledCol] : 0;
        Bs[ty * blockDim.x + tx] = (tiledRow < n && col < n) ? B[tiledRow * n + col] : 0;

        __syncthreads();

        int limit = min(blockDim.x, n - t * blockDim.x);
        for (int k = 0; k < limit; k++)
            sum += As[ty * blockDim.x + k] * Bs[k * blockDim.x + tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

template <typename T>
void launch_matmul(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim)
{
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    size_t smem = 2 * block_dim * block_dim * sizeof(T);

    matmul_kernel<<<grid, block, smem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

// These functions match matmul.cuh exactly
void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
    launch_matmul(A, B, C, n, block_dim);
}

void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    launch_matmul(A, B, C, n, block_dim);
}

void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    launch_matmul(A, B, C, n, block_dim);
}
