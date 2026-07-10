#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Reverse-engineer address->slice hash on H800. A group-A SM is a slice detector
// (fast=home slice, slow=other). Probe base and base^(1<<b) for each bit b; if
// latency class flips, bit b is in the (XOR) hash. Also test bit pairs to confirm
// linearity. ld.cg single-line dependent-chase latency.
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
#define CHASE 2048
#define MAXO 64
__device__ uint32_t g_lat[512*MAXO]; __device__ unsigned g_sm[512];
__device__ __forceinline__ uint32_t chase1(char* p){
    uint64_t off=0,t0,t1;
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)(p+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1));
    if(off==0xdead)*(volatile uint64_t*)p=off;
    return (uint32_t)((t1-t0)/CHASE);
}
__global__ void setup(char* buf, unsigned long long* offs, int n){
    // each probed line self-points (off stays 0)
    for(int i=0;i<n;i++) *(uint64_t*)(buf+offs[i])=0;
}
__global__ void probe(char* buf, unsigned long long* offs, int n){
    if(threadIdx.x) return; unsigned sm=smid(); uint32_t lat[MAXO];
    for(int i=0;i<n;i++) lat[i]=chase1(buf+offs[i]);
    if(atomicCAS(&g_sm[sm],0xffffffffu,sm)==0xffffffffu)
        for(int i=0;i<n;i++) g_lat[sm*MAXO+i]=lat[i];
}
int main(){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    // offsets: 0, then 1<<b for b=7..34, then a few pairs to check linearity
    unsigned long long offs[MAXO]; int n=0;
    offs[n++]=0;
    for(int b=7;b<=34;b++) offs[n++]=(1ULL<<b);
    // pairs (to verify XOR linearity): (1<<a)^(1<<c)
    int pa[]={8,9,10,11,12,8,9}, pc[]={9,10,11,12,13,20,25};
    int npair=7;
    for(int k=0;k<npair;k++) offs[n++]=(1ULL<<pa[k])|(1ULL<<pc[k]);
    printf("H800 SMs=%d nprobe=%d\n",SMs,n);
    size_t bytes=(1ULL<<35)+ (1<<20);   // 32 GB to cover bit 34
    char* buf; cudaError_t me=cudaMalloc(&buf,bytes);
    if(me){printf("malloc fail %s\n",cudaGetErrorString(me));return 1;}
    cudaMemset(buf,0,bytes);
    unsigned long long* doff; cudaMalloc(&doff,n*8); cudaMemcpy(doff,offs,n*8,cudaMemcpyHostToDevice);
    setup<<<1,1>>>(buf,doff,n);
    unsigned fss[512]; for(int i=0;i<512;i++)fss[i]=0xffffffff; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss));
    cudaDeviceSynchronize();
    probe<<<SMs*3,1>>>(buf,doff,n); cudaError_t e=cudaDeviceSynchronize();
    uint32_t lat[512*MAXO]; unsigned sm[512];
    cudaMemcpyFromSymbol(lat,g_lat,sizeof(lat)); cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    printf("err=%s\n",e?cudaGetErrorString(e):"ok");
    // header
    printf("SM   base "); for(int b=7;b<=34;b++)printf("b%-4d",b);
    for(int k=0;k<npair;k++)printf("%d^%d  ",pa[k],pc[k]); printf("\n");
    for(int s=0;s<SMs;s++){ if(sm[s]==0xffffffff)continue;
        printf("%-4d ",s); for(int i=0;i<n;i++)printf("%-5u",lat[s*MAXO+i]); printf("\n");

    }
    return 0;
}
