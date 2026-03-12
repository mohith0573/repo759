#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include "count.cuh"

int main(int argc,char* argv[])
{
    int n=atoi(argv[1]);

    thrust::host_vector<int> h(n);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0,500);

    for(int i=0;i<n;i++)
        h[i]=dist(gen);

    thrust::device_vector<int> d=h;

    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    count(d,values,counts);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    std::cout<<values.back()<<std::endl;
    std::cout<<counts.back()<<std::endl;
    std::cout<<ms<<std::endl;
}
