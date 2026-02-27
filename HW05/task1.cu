#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <type_traits>
#include "matmul.cuh"

template <typename T>
void run_test(unsigned int n, unsigned int block_dim,
              void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int))
{
    T *A, *B, *C;
    cudaMallocManaged(&A, n*n*sizeof(T));
    cudaMallocManaged(&B, n*n*sizeof(T));
    cudaMallocManaged(&C, n*n*sizeof(T));

    for (unsigned int i = 0; i < n*n; i++) {
        if constexpr (std::is_integral<T>::value)
            A[i] = rand() % 100;
        else
            A[i] = (rand() % 200) / 100.0 - 1;

        if constexpr (std::is_integral<T>::value)
            B[i] = rand() % 100;
        else
            B[i] = (rand() % 200) / 100.0 - 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_func(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << C[0] << "\n" << C[n*n-1] << "\n" << ms << "\n";

    cudaFree(A); cudaFree(B); cudaFree(C);
}

int main(int argc, char** argv)
{
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
