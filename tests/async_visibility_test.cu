#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Probe: does data become visible to a generic LDS at mbarrier completion, or is
// a fence.proxy.async required (=> async staging buffer, completion != landed)?
// Pre-fill shared with SENTINEL. TMA-copy fresh data. mbarrier-wait. Read WITHOUT
// the proxy fence (mode 0) or WITH it (mode 1). Count stale reads (==sentinel).
#define SENT 0xDEAD1234u
#define N 3072   // words per CTA

template<int USE_FENCE> __global__ void probe(const uint32_t* gsrc, uint32_t* stale_count){
    extern __shared__ uint32_t smem[];
    __shared__ uint64_t mbar;
    uint32_t smem_a=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_a=(uint32_t)__cvta_generic_to_shared(&mbar);
    // pre-fill with sentinel via generic stores
    for(int i=threadIdx.x;i<N;i+=blockDim.x) smem[i]=SENT;
    if(threadIdx.x==0) asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(mbar_a):"memory");
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;":::"memory");   // make sentinel writes visible to async proxy
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
            ::"r"(smem_a),"l"(gsrc),"r"(N*4),"r"(mbar_a):"memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mbar_a),"r"(N*4):"memory");
    }
    __syncthreads();
    // wait for completion
    uint32_t done=0;
    while(!done) asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],0; selp.u32 %0,1,0,p;}"
        :"=r"(done):"r"(mbar_a):"memory");
    if(USE_FENCE) asm volatile("fence.proxy.async.shared::cta;":::"memory");
    // read back immediately (generic proxy). Data is all 0x42..., sentinel is 0xDEAD1234
    int stale=0;
    for(int i=threadIdx.x;i<N;i+=blockDim.x){
        uint32_t v; asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(smem_a+i*4):"memory");
        if(v==SENT) stale++;
    }
    if(stale) atomicAdd(stale_count,stale);
}
int main(int argc,char**argv){
    int use_fence=argc>1?atoi(argv[1]):0;
    int blocks=4096, iters=200;
    uint32_t* g; cudaMalloc(&g,N*4); cudaMemset(g,0x42,N*4);  // fresh data != sentinel
    uint32_t* sc; cudaMalloc(&sc,4);
    unsigned long long total=0;
    for(int it=0;it<iters;it++){
        cudaMemset(sc,0,4);
        if(use_fence) probe<1><<<blocks,256,N*4>>>(g,sc); else probe<0><<<blocks,256,N*4>>>(g,sc);
        uint32_t h; cudaMemcpy(&h,sc,4,cudaMemcpyDeviceToHost); total+=h;
    }
    cudaError_t e=cudaDeviceSynchronize();
    printf("use_fence=%d  stale_reads=%llu / %d launches  (%s)\n",use_fence,total,blocks*iters,cudaGetErrorString(e));
    return 0;
}
