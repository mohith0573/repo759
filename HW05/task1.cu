#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include "matmul.cuh"

// -----------------------
// Tiled matrix multiplication kernel
// -----------------------
template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n) {
    extern __shared__ unsigned char smem[];
    T* As = (T*)smem;
    T* Bs = (T*)&As[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    T sum = 0;

    for (int t = 0; t < (n + blockDim.x - 1) / blockDim.x; t++) {
        int tiledCol = t * blockDim.x + tx;
        int tiledRow = t * blockDim.y + ty;

        As[ty * blockDim.x + tx] = (row < n && tiledCol < n) ? A[row * n + tiledCol] : 0;
        Bs[ty * blockDim.x + tx] = (tiledRow < n && col < n) ? B[tiledRow * n + col] : 0;

        __syncthreads();

        for (int k = 0; k < blockDim.x; k++)
            sum += As[ty * blockDim.x + k] * Bs[k * blockDim.x + tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// -----------------------
// Launcher
// -----------------------
template <typename T>
void launch_matmul(const T* dA, const T* dB, T* dC, unsigned int n, unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t smem = 2 * block_dim * block_dim * sizeof(T);

    matmul_kernel<<<grid, block, smem>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();
}

// -----------------------
// Wrapper functions for matmul.cuh
// -----------------------
void matmul_1(const int* A, const int* B, int* C, unsigned int n, unsigned int block_dim) {
    launch_matmul(A, B, C, n, block_dim);
}
void matmul_2(const float* A, const float* B, float* C, unsigned int n, unsigned int block_dim) {
    launch_matmul(A, B, C, n, block_dim);
}
void matmul_3(const double* A, const double* B, double* C, unsigned int n, unsigned int block_dim) {
    launch_matmul(A, B, C, n, block_dim);
}

// -----------------------
// Test runner
// -----------------------
template <typename T>
void run_test(unsigned int n, unsigned int block_dim,
              void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int)) {

    // Host arrays
    T* hA = new T[n * n];
    T* hB = new T[n * n];
    T* hC = new T[n * n];

    // Initialize random values [-1, 1]
    for (unsigned int i = 0; i < n * n; i++) {
        hA[i] = static_cast<T>((rand() % 200) / 100.0 - 1);
        hB[i] = static_cast<T>((rand() % 200) / 100.0 - 1);
    }

    // Device arrays
    T *dA, *dB, *dC;
    cudaMalloc(&dA, n * n * sizeof(T));
    cudaMalloc(&dB, n * n * sizeof(T));
    cudaMalloc(&dC, n * n * sizeof(T));

    // Copy to device
    cudaMemcpy(dA, hA, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * n * sizeof(T), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_func(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cudaMemcpy(hC, dC, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    // Print first, last elements + time
    std::cout << hC[0] << "\n" << hC[n * n - 1] << "\n" << ms << "\n";

    // Cleanup
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);

    run_test<int>(n, block_dim, matmul_1);
    run_test<float>(n, block_dim, matmul_2);
    run_test<double>(n, block_dim, matmul_3);

    return 0;
}
