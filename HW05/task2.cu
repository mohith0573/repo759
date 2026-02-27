#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include "reduce.cuh"

int main(int argc, char** argv)
{
    if(argc < 3){
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return 1;
    }

    unsigned int N = atoi(argv[1]);
    unsigned int threads = atoi(argv[2]);

    // Allocate host array
    float *h = new float[N];
    for(unsigned int i = 0; i < N; i++)
        h[i] = (rand() % 200) / 100.0f - 1.0f;  // values in [-1,1]

    // Allocate device arrays
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    unsigned int blocks = (N + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&d_out, blocks * sizeof(float));

    cudaMemcpy(d_in, h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(&d_in, &d_out, N, threads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float result;
    cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    float ms; 
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << result << "\n" << ms << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h;

    return 0;
}
