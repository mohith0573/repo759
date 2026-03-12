#define CUB_STDERR
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
#include <random>
#include <cuda_runtime.h>

using namespace cub;

CachingDeviceAllocator g_allocator(true);

int main(int argc, char* argv[]) {

    int n = atoi(argv[1]);

    float* h_in = (float*)malloc(sizeof(float)*n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0,1.0);

    for(int i=0;i<n;i++)
        h_in[i] = dist(gen);

    float *d_in=NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in,sizeof(float)*n));

    CubDebugExit(cudaMemcpy(d_in,h_in,sizeof(float)*n,cudaMemcpyHostToDevice));

    float *d_sum=NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum,sizeof(float)));

    void *d_temp_storage=NULL;
    size_t temp_storage_bytes=0;

    DeviceReduce::Sum(d_temp_storage,temp_storage_bytes,d_in,d_sum,n);
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage,temp_storage_bytes));

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    DeviceReduce::Sum(d_temp_storage,temp_storage_bytes,d_in,d_sum,n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    float result;
    cudaMemcpy(&result,d_sum,sizeof(float),cudaMemcpyDeviceToHost);

    printf("%f\n",result);
    printf("%f\n",ms);

    if(d_in) g_allocator.DeviceFree(d_in);
    if(d_sum) g_allocator.DeviceFree(d_sum);
    if(d_temp_storage) g_allocator.DeviceFree(d_temp_storage);

    free(h_in);
}
