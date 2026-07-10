#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Where are global atomics performed? Latency of a DEPENDENT atomic chain by
// scope. Each atom.add's added value depends on the previous returned value
// (r = atom.add(p, r&1)), so each must round-trip before the next issues.
// If .cta << .gpu, .cta global atomics execute at L1 and .gpu at L2.
#define K 20000
template<int SCOPE>
__device__ __forceinline__ uint32_t aadd(uint32_t* p, uint32_t v){
    uint32_t r;
    if(SCOPE==0) asm volatile("atom.relaxed.cta.global.add.u32 %0,[%1],%2;":"=r"(r):"l"(p),"r"(v):"memory");
    else if(SCOPE==1) asm volatile("atom.relaxed.gpu.global.add.u32 %0,[%1],%2;":"=r"(r):"l"(p),"r"(v):"memory");
    else asm volatile("atom.relaxed.sys.global.add.u32 %0,[%1],%2;":"=r"(r):"l"(p),"r"(v):"memory");
    return r;
}
template<int SCOPE> __global__ void chain(uint32_t* p, uint64_t* out){
    uint32_t r=1; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<K;i++){ r = aadd<SCOPE>(p, (r&1)+1); }
    uint64_t t1=clock64(); if(threadIdx.x==0){ out[0]=t1-t0; out[1]=r; }
}
// shared atomic chain for reference (ATOMS, done in SM shared unit)
__global__ void chain_shared(uint64_t* out){
    __shared__ uint32_t s; if(threadIdx.x==0)s=1; __syncthreads();
    uint32_t a=(uint32_t)__cvta_generic_to_shared(&s); uint32_t r=1; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<K;i++){ asm volatile("atom.relaxed.cta.shared.add.u32 %0,[%1],%2;":"=r"(r):"r"(a),"r"((r&1)+1):"memory"); }
    uint64_t t1=clock64(); if(threadIdx.x==0){ out[0]=t1-t0; out[1]=r; }
}
int main(){
    uint32_t* p; uint64_t* o,h[2]; cudaMalloc(&p,4); cudaMalloc(&o,16);
    auto run=[&](const char*n,auto k){ cudaMemset(p,0,4); k(); cudaDeviceSynchronize();
        cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost); printf("%-16s %.1f cyc/atomic\n",n,(double)h[0]/K); };
    run("shared .cta",  [&]{ chain_shared<<<1,1>>>(o); });
    run("global .cta",  [&]{ chain<0><<<1,1>>>(p,o); });
    run("global .gpu",  [&]{ chain<1><<<1,1>>>(p,o); });
    run("global .sys",  [&]{ chain<2><<<1,1>>>(p,o); });
    return 0;
}
