#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "scan.cuh"

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr<<"Usage: ./task2 n threads_per_block\n";
        return 1;
    }

    unsigned int n = atoi(argv[1]);
    unsigned int threads = atoi(argv[2]);

    float *input,*output;

    cudaMallocManaged(&input,n*sizeof(float));
    cudaMallocManaged(&output,n*sizeof(float));

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    for(unsigned int i=0;i<n;i++)
        input[i] = dist(gen);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    scan(input,output,n,threads);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    std::cout << output[n-1] << std::endl;
    std::cout << ms << std::endl;

    cudaFree(input);
    cudaFree(output);

    return 0;
}
