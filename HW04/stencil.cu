#include "stencil.cuh"
#include <cuda_runtime.h>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R)
{
    extern __shared__ float s[];

    float* s_mask = s;
    float* s_img  = s + (2*R+1);
    float* s_out  = s_img + blockDim.x + 2*R;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if(tid < 2*R+1)
        s_mask[tid] = mask[tid];

    int start = blockIdx.x * blockDim.x - R;
    for(int i=tid;i<blockDim.x+2*R;i+=blockDim.x){
        int idx = start + i;
        s_img[i] = (idx<0 || idx>=n)? 1.0f : image[idx];
    }

    __syncthreads();

    if(gid<n){
        float sum=0;
        for(int j=-R;j<=R;j++)
            sum += s_img[tid+j+R]*s_mask[j+R];
        s_out[tid]=sum;
    }

    __syncthreads();

    if(gid<n)
        output[gid]=s_out[tid];
}

__host__ void stencil(const float* image,const float* mask,float* output,
                      unsigned int n,unsigned int R,unsigned int tpb)
{
    unsigned int blocks=(n+tpb-1)/tpb;
    size_t shmem=(2*R+1 + tpb+2*R + tpb)*sizeof(float);

    stencil_kernel<<<blocks,tpb,shmem>>>(image,mask,output,n,R);
}
