#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mmul.h"

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "Usage: ./task1 n n_tests\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int n_tests = atoi(argv[2]);

    size_t size = n * n * sizeof(float);

    float *A, *B, *C;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    for(int i=0;i<n*n;i++)
    {
        A[i] = dist(gen);
        B[i] = dist(gen);
        C[i] = dist(gen);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i=0;i<n_tests;i++)
        mmul(handle,A,B,C,n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    std::cout << ms/n_tests << std::endl;

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
