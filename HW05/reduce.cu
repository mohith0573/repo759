#include <cuda_runtime.h>
#include "reduce.cuh"

__global__ void reduce_kernel(double *g_idata, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    double sum = 0.0;
    if(i < n) sum = g_idata[i];
    if(i + blockDim.x < n) sum += g_idata[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

void reduce(double **input, double **output, unsigned int N, unsigned int threads)
{
    unsigned int blocks = (N + threads*2 - 1) / (threads*2);

    reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(*input, *output, N);
    cudaDeviceSynchronize();

    double **in_ptr = output;

    while(blocks > 1) {
        unsigned int newBlocks = (blocks + threads*2 - 1) / (threads*2);
        reduce_kernel<<<newBlocks, threads, threads * sizeof(double)>>>(*in_ptr, *in_ptr, blocks);
        cudaDeviceSynchronize();
        blocks = newBlocks;
    }
}
