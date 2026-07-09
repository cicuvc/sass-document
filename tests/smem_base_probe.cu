#include <cstdint>

// (1) direct shared access — offset-based, likely no const base
__global__ void s_direct(int *out)
{
    __shared__ int smem[256];
    smem[threadIdx.x] = threadIdx.x;
    __syncthreads();
    out[threadIdx.x] = smem[255 - threadIdx.x];
}

// (2) generic pointer to shared — cvta.shared needs the shared window base
__global__ void s_generic(int *out, int sel)
{
    __shared__ int smem[256];
    int *p = smem;                 // decays; taking generic address
    volatile int *gp = (volatile int *)(p + (sel & 255));
    *gp = sel;                     // generic ST -> needs shared window base
    __syncthreads();
    out[threadIdx.x] = *gp;
}

// (3) dynamic extern shared — also generic-ish base
extern __shared__ int dsmem[];
__global__ void s_dyn(int *out, int sel)
{
    dsmem[threadIdx.x] = sel;
    __syncthreads();
    out[threadIdx.x] = dsmem[(sel + threadIdx.x) & 63];
}
