#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// L2-coherence probe. Producer (block0/SM_A) increments a global counter; consumer
// (block1/SM_B, different SM) spin-reads. Modes: 0=relaxed.gpu (STRONG.GPU, no
// per-iter L1 inval), 1=weak (LDG.E, may cache L1), 2=acquire.gpu (STRONG.GPU +
// CCTL.IVALL). Hoist-proof address (runtime mask). If relaxed.gpu tracks the
// producer live+monotonic without per-iter invalidation, L2 is the single coherent
// point (STRONG.GPU ops reach it). If weak gets stuck, L1 caches stale (per-SM).
#define N 2000000
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
extern "C" __global__ void probe(uint32_t* flag, uint32_t* out, uint32_t mask, int mode){
    if(blockIdx.x==0 && threadIdx.x==0){                 // producer
        out[10]=smid();
        for(uint32_t j=1;j<=N;j++) asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(flag),"r"(j):"memory");
    } else if(blockIdx.x==1 && threadIdx.x==0){          // consumer
        out[11]=smid();
        uint32_t maxv=0,last=0; unsigned long long backward=0; uint32_t spin=0;
        while(true){
            uint32_t off=(spin&mask); uint32_t* p=flag+off; uint32_t v;
            if(mode==0)      asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");
            else if(mode==1) asm volatile("ld.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");
            else { asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory"); }
            if(v<last) backward++;
            last=v; if(v>maxv)maxv=v;
            if(v>=N) break;
            if(++spin>800000000u) break;
        }
        out[0]=maxv; out[1]=(uint32_t)backward; out[2]=(spin>800000000u);
    }
}
int main(int argc,char**argv){
    int mode=argc>1?atoi(argv[1]):0;
    uint32_t *flag,*out,h[16];
    cudaMalloc(&flag,64); cudaMemset(flag,0,64); cudaMalloc(&out,64); cudaMemset(out,0,64);
    probe<<<2,1>>>(flag,out,0u,mode);
    cudaError_t e=cudaDeviceSynchronize(); cudaMemcpy(h,out,64,cudaMemcpyDeviceToHost);
    const char* mn[]={"relaxed.gpu","weak(LDG.E)","acquire.gpu"};
    printf("mode=%-12s maxSeen=%u backward=%u stuck=%u | producerSM=%u consumerSM=%u %s\n",
        mn[mode],h[0],h[1],h[2],h[10],h[11], e?cudaGetErrorString(e):"");
    return 0;
}
