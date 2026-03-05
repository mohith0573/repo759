#include "mmul.h"
#include <cuda_runtime.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n)
{
    const float alpha = 1.0f;
    const float beta  = 1.0f;

    // C := A*B + C
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, n, n,
                &alpha,
                A, n,
                B, n,
                &beta,
                C, n);

    cudaDeviceSynchronize();
}
