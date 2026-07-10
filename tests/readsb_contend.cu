#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// (a) vs (b): does arbiter/MIO contention from OTHER subcores lengthen the
// timed warp's store read-scoreboard hold? Warp0 tid0 runs the reuse loop and
// times it; the rest of the CTA floods global stores to congest the LSU/arbiter.
#define K 100000
__global__ void contend(uint32_t* p, uint32_t* flood, uint32_t din, uint64_t* out, int doFlood){
    int tid=threadIdx.x;
    if(tid==0){
        uint32_t d=din; uint64_t t0=clock64();
        #pragma unroll 1
        for(int i=0;i<K;i++){ asm volatile("st.relaxed.cta.global.u32 [%1], %0;"::"r"(d),"l"(p):"memory"); d+=1; }
        uint64_t t1=clock64(); out[0]=t1-t0; out[1]=d;
    } else if(doFlood){
        // flood: independent stores hammering the memory pipe / arbiter
        uint32_t* q=flood + tid*64;
        for(int i=0;i<K*4;i++){ asm volatile("st.relaxed.cta.global.u32 [%0], %1;"::"l"(q+(i&63)),"r"(i):"memory"); }
    }
}
int main(){ cudaFree(0);
    uint64_t *o,h[2]; uint32_t *p,*flood;
    cudaMalloc(&o,16); cudaMalloc(&p,4096); cudaMalloc(&flood,sizeof(uint32_t)*1024*64);
    for(int fl=0; fl<2; ++fl){
        contend<<<1,256>>>(p,flood,7,o,fl);
        cudaError_t e=cudaDeviceSynchronize(); cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);
        printf("reuse-loop cyc/iter = %.2f   (%s arbiter flood)  %s\n",(double)h[0]/K, fl?"WITH":"NO", e?cudaGetErrorString(e):"");
    }
    return 0;
}
