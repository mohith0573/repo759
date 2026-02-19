#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char* argv[])
{
    size_t n = std::stoul(argv[1]);
    unsigned int tpb = std::stoul(argv[2]);
    size_t bytes = n * n * sizeof(float);

    float *hA = new float[n*n];
    float *hB = new float[n*n];
    float *hC = new float[n*n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    for(size_t i=0;i<n*n;i++){
        hA[i]=dist(gen);
        hB[i]=dist(gen);
    }

    float *dA,*dB,*dC;
    cudaMalloc(&dA,bytes);
    cudaMalloc(&dB,bytes);
    cudaMalloc(&dC,bytes);

    cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(dA,dB,dC,n,tpb);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(hC,dC,bytes,cudaMemcpyDeviceToHost);

    std::cout<<hC[n*n-1]<<"\n";
    std::cout<<ms<<"\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
}
