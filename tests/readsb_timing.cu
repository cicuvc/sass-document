#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Measure store read-scoreboard HOLD time: loop body = STG [p],Rk ; Rk+=1.
// In a loop Rk is a fixed physical reg, so each iter's ADD (writes Rk) must wait
// until the STG has read Rk => iteration time ~ read-SB release latency.
// Compare global vs shared: if equal & small => read early (excludes c); if
// global>>shared => register held to completion (c).
#define K 100000

__global__ void loop_stg(uint32_t* p, uint32_t din, uint64_t* out){
    uint32_t d=din; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<K;i++){ asm volatile("st.relaxed.cta.global.u32 [%1], %0;"::"r"(d),"l"(p):"memory"); d+=1; }
    uint64_t t1=clock64(); if(threadIdx.x==0){out[0]=t1-t0; out[1]=d;}
}
__global__ void loop_sts(uint32_t din, uint64_t* out){
    __shared__ uint32_t s[4]; uint32_t a=(uint32_t)__cvta_generic_to_shared(&s[0]);
    uint32_t d=din; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<K;i++){ asm volatile("st.relaxed.cta.shared.u32 [%1], %0;"::"r"(d),"r"(a):"memory"); d+=1; }
    uint64_t t1=clock64(); if(threadIdx.x==0){out[0]=t1-t0; out[1]=d;}
}
__global__ void loop_add(uint32_t din, uint64_t* out){   // ALU floor: just the recurrence
    uint32_t d=din; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<K;i++){ asm volatile("add.u32 %0,%0,1;":"+r"(d)::"memory"); }
    uint64_t t1=clock64(); if(threadIdx.x==0){out[0]=t1-t0; out[1]=d;}
}
// control: store but NO reuse (data reg constant) -> no read-SB WAR; pure issue rate
__global__ void loop_stg_noreuse(uint32_t* p, uint32_t din, uint64_t* out){
    uint32_t d=din; uint64_t t0=clock64();
    #pragma unroll 1
    for(int i=0;i<K;i++){ asm volatile("st.relaxed.cta.global.u32 [%1], %0;"::"r"(d),"l"(p):"memory"); }
    uint64_t t1=clock64(); if(threadIdx.x==0){out[0]=t1-t0; out[1]=d;}
}
template<class F> double t(F kern){
    uint64_t *o,h[2]; uint32_t* p; cudaMalloc(&o,16); cudaMalloc(&p,4096);
    kern(p,o); cudaDeviceSynchronize(); cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);
    cudaFree(o);cudaFree(p); return (double)h[0]/K;
}
int main(){ cudaFree(0);
    uint64_t *o,h[2]; uint32_t* p; cudaMalloc(&o,16); cudaMalloc(&p,4096);
    auto R=[&](const char* n, auto launch){ launch(); cudaError_t e=cudaDeviceSynchronize();
        cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost); printf("%-22s cyc/iter=%.2f  (d=%u) %s\n",n,(double)h[0]/K,(uint32_t)h[1],e?cudaGetErrorString(e):""); };
    R("ALU add (floor)",      [&]{ loop_add<<<1,1>>>(7,o);});
    R("STG global +reuse",    [&]{ loop_stg<<<1,1>>>(p,7,o);});
    R("STS shared +reuse",    [&]{ loop_sts<<<1,1>>>(7,o);});
    R("STG global no-reuse",  [&]{ loop_stg_noreuse<<<1,1>>>(p,7,o);});
    cudaFree(o);cudaFree(p); return 0;
}
