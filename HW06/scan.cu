#include "scan.cuh"
#include <cuda_runtime.h>

__global__
void hillis_steele(const float* input, float* output, unsigned int n, unsigned int stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
    {
        if(i >= stride)
            output[i] = input[i] + input[i - stride];
        else
            output[i] = input[i];
    }
}

__host__
void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block)
{
    float *temp1, *temp2;

    cudaMallocManaged(&temp1, n*sizeof(float));
    cudaMallocManaged(&temp2, n*sizeof(float));

    for(unsigned int i=0;i<n;i++)
        temp1[i] = input[i];

    int blocks = (n + threads_per_block -1)/threads_per_block;

    for(unsigned int stride=1; stride < n; stride*=2)
    {
        hillis_steele<<<blocks,threads_per_block>>>(temp1,temp2,n,stride);
        cudaDeviceSynchronize();

        float* swap = temp1;
        temp1 = temp2;
        temp2 = swap;
    }

    for(unsigned int i=0;i<n;i++)
        output[i] = temp1[i];

    cudaFree(temp1);
    cudaFree(temp2);
}
