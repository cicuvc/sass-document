#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Single-line L2 latency per SM, swept over fine address offsets. Each "region"
// is ONE self-pointing line (stays on ONE slice). ld.cg -> L2 (bypass L1). If 2
// NUMA slices: for a FIXED offset, per-SM latency is bimodal (near/far SM groups);
// and sweeping offset by the slice-select stride flips near<->far.
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
#define CHASE 2048
#define NREG 16
__device__ uint32_t g_lat[512*NREG]; __device__ unsigned g_sm[512];
__device__ __forceinline__ uint32_t chase1(uint64_t* line){
    uint64_t off=0,t0,t1;
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)((char*)line+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1));
    if(off==0xdead) line[0]=off;
    return (uint32_t)((t1-t0)/CHASE);
}
__global__ void probe(char* base, long long stride){
    if(threadIdx.x) return; unsigned sm=smid(); uint32_t lat[NREG];
    for(int r=0;r<NREG;r++) lat[r]=chase1((uint64_t*)(base+(long long)r*stride));
    if(atomicCAS(&g_sm[sm],0xffffffffu,sm)==0xffffffffu)
        for(int r=0;r<NREG;r++) g_lat[sm*NREG+r]=lat[r];
}
int main(int argc,char**argv){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    long long stride=argc>1?atoll(argv[1]):2048;   // bytes between probed lines
    printf("H800 SMs=%d line-stride=%lldB\n",SMs,stride);
    size_t bytes=(size_t)stride*NREG+ (1<<20);
    char* buf; cudaMalloc(&buf,bytes); cudaMemset(buf,0,bytes); cudaDeviceSynchronize();
    unsigned fss[512]; for(int i=0;i<512;i++)fss[i]=0xffffffff; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss));
    probe<<<SMs*3,1>>>(buf,stride); cudaError_t e=cudaDeviceSynchronize();
    uint32_t lat[512*NREG]; unsigned sm[512];
    cudaMemcpyFromSymbol(lat,g_lat,sizeof(lat)); cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    printf("err=%s\n",e?cudaGetErrorString(e):"ok");
    printf("SM  "); for(int r=0;r<NREG;r++)printf("%-5d",r); printf("\n");
    for(int s=0;s<SMs;s++){ if(sm[s]==0xffffffff)continue;
        printf("%-4d",s); for(int r=0;r<NREG;r++)printf("%-5u",lat[s*NREG+r]); printf("\n"); }
    return 0;
}
