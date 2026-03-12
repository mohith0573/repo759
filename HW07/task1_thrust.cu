#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {

    int n = atoi(argv[1]);

    thrust::host_vector<float> h_vec(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0,1.0);

    for(int i=0;i<n;i++)
        h_vec[i] = dist(gen);

    thrust::device_vector<float> d_vec = h_vec;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    std::cout << result << std::endl;
    std::cout << ms << std::endl;
}
