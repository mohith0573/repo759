#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include "matmul.cuh"

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

    // Call functions already defined in matmul.cu
    run_test<int>(n, block_dim, matmul_1);
    run_test<float>(n, block_dim, matmul_2);
    run_test<double>(n, block_dim, matmul_3);

    return 0;
}
