#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Flood from a DIFFERENT SM (separate stream / kernel) to isolate arbiter
// contention from per-SM issue contention. Timed warp runs alone in its SM,
// while a persistent flood kernel hammers the L1/LSU from other SMs.
__global__ void timer(uint32_t* p, uint64_t* out){
    if(threadIdx.x!=0) return;
    uint32_t d=7; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<100000;i++){ asm volatile("st.relaxed.cta.global.u32 [%1], %0;"::"r"(d),"l"(p):"memory"); d+=1; }
    uint64_t t1=clock64(); out[0]=t1-t0; out[1]=d;
}
// flood stays resident, hammers stores from many blocks
__device__ volatile int stop_signal;
__global__ void flood(uint32_t* f){ int tid=threadIdx.x+blockIdx.x*blockDim.x; for(int i=0;!stop_signal;i++){ asm volatile("st.relaxed.cta.global.u32 [%0], %1;"::"l"(f+((tid+i)&0xFFFF)),"r"(i):"memory"); }}
int main(){ cudaFree(0);
    uint64_t*o; uint32_t*p,*f; 
    cudaMalloc(&o,16); cudaMalloc(&p,4096); cudaMalloc(&f,sizeof(uint32_t)*65536);
    // baseline
    timer<<<1,1>>>(p,o); cudaDeviceSynchronize();
    uint64_t h[2]; cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);
    printf("NO flood   cyc/iter=%.2f\n",(double)h[0]/100000);
    // with flood kernel running concurrently
    cudaMemset(o,0,16);
    int nstops=0; cudaMemcpyToSymbol(stop_signal,&nstops,4);
    cudaStream_t s1,s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    flood<<<200,128,0,s1>>>(f);
    timer<<<1,1,0,s2>>>(p,o);
    cudaStreamSynchronize(s2);
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);
    nstops=1; cudaMemcpyToSymbol(stop_signal,&nstops,4);
    cudaStreamSynchronize(s1);
    printf("WITH flood (other SMs)  cyc/iter=%.2f\n",(double)h[0]/100000);
    cudaFree(o);cudaFree(p);cudaFree(f); return 0;
}
