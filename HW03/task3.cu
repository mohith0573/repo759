#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./task3 n\n";
        return 1;
    }

    unsigned int n = std::stoul(argv[1]);

    float *hA = new float[n];
    float *hB = new float[n];

    // Random generators
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
    std::uniform_real_distribution<float> distB(0.0f, 1.0f);

    for (unsigned int i = 0; i < n; ++i)
    {
        hA[i] = distA(gen);
        hB[i] = distB(gen);
    }

    float *dA, *dB;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));

    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vscale<<<blocks, threadsPerBlock>>>(dA, dB, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << milliseconds << std::endl;
    std::cout << hB[0] << std::endl;
    std::cout << hB[n - 1] << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    delete[] hA;
    delete[] hB;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

