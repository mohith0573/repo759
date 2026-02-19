#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc,char* argv[])
{
    unsigned int n=std::stoul(argv[1]);
    unsigned int R=std::stoul(argv[2]);
    unsigned int tpb=std::stoul(argv[3]);

    float *hImg=new float[n];
    float *hMask=new float[2*R+1];
    float *hOut=new float[n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1,1);

    for(unsigned i=0;i<n;i++) hImg[i]=dist(gen);
    for(unsigned i=0;i<2*R+1;i++) hMask[i]=dist(gen);

    float *dImg,*dMask,*dOut;
    cudaMalloc(&dImg,n*sizeof(float));
    cudaMalloc(&dMask,(2*R+1)*sizeof(float));
    cudaMalloc(&dOut,n*sizeof(float));

    cudaMemcpy(dImg,hImg,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dMask,hMask,(2*R+1)*sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(dImg,dMask,dOut,n,R,tpb);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(hOut,dOut,n*sizeof(float),cudaMemcpyDeviceToHost);

    std::cout<<hOut[n-1]<<"\n";
    std::cout<<ms<<"\n";
}
