#include <iostream>
#include <cuda_runtime.h>
#include <random>

__global__ void computeKernel(int *dA, int a)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int idx = y * blockDim.x + x;

    dA[idx] = a * x + y;
}

int main()
{
    const int N = 16;

    int *dA;
    int hA[N];

    cudaMalloc(&dA, N * sizeof(int));

    // random integer between -100 and 100
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-100, 100);
    int a = dist(gen);

    computeKernel<<<2, 8>>>(dA, a);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << hA[i] << "\n ";
    std::cout << std::endl;

    cudaFree(dA);
    return 0;
}

