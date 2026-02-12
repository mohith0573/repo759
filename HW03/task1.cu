#include <iostream>
#include <cuda_runtime.h>

__global__ void factorialKernel(int *dA)
{
    int a = threadIdx.x;   // since only 1 block
    int val = 1;

    for (int i = 1; i <= a + 1; ++i)
        val *= i;

    dA[a] = val;
}

int main()
{
    const int N = 8;

    int *dA;
    int hA[N];

    cudaMalloc(&dA, N * sizeof(int));

    factorialKernel<<<1, 8>>>(dA);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << hA[i] << std::endl;

    cudaFree(dA);
    return 0;
}

