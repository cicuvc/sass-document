// Hoist-proof coherence probe. Address = flag + (spin & mask) with mask=0 passed
// at runtime, so ptxas cannot prove the load loop-invariant => LDS/LDG stays in
// the loop and actually re-fetches each iteration. Now staleness (if any) is
// hardware, not compiler LICM.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define N 2000000

template<int WEAKLD,int SHARED>
__global__ void coh(uint32_t* g, uint32_t* out, uint32_t mask /*=0*/){
    __shared__ uint32_t s[2];
    int tid=threadIdx.x, warp=tid>>5;
    if(tid==0){ if(SHARED) s[0]=0; else g[0]=0; }
    __syncthreads();
    if(warp==0 && tid==0){
        for(uint32_t j=1;j<=N;j++){
            if(SHARED){ uint32_t a=(uint32_t)__cvta_generic_to_shared(&s[0]); asm volatile("st.relaxed.cta.shared.u32 [%0],%1;"::"r"(a),"r"(j):"memory"); }
            else asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(g),"r"(j):"memory");
        }
    } else if(warp==1 && tid==32){
        uint32_t maxv=0,last=0; unsigned long long backward=0,reads=0; uint32_t spin=0;
        while(true){
            uint32_t off = (spin & mask);          // ==0 at runtime, opaque to ptxas
            uint32_t v;
            if(SHARED){ uint32_t a=(uint32_t)__cvta_generic_to_shared(&s[0])+off*4; if(WEAKLD) asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory"); else asm volatile("ld.relaxed.cta.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory"); }
            else { uint32_t* p=g+off; if(WEAKLD) asm volatile("ld.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory"); else asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory"); }
            reads++;
            if(v<last) backward++;
            last=v; if(v>maxv)maxv=v;
            if(v>=N) break;
            spin++;
            if(spin>500000000u) break;             // stuck guard
        }
        out[0]=maxv; out[1]=(uint32_t)backward; out[2]=(uint32_t)(reads&0xffffffff); out[3]=spin>=500000000u?1:0;
    }
}
template<int W,int S> void run(const char* name){
    uint32_t *g,*out,h[4];
    cudaMalloc(&g,64); cudaMalloc(&out,16); cudaMemset(out,0,16);
    coh<W,S><<<1,64>>>(g,out,0u);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(h,out,16,cudaMemcpyDeviceToHost);
    printf("%-30s maxSeen=%u backward=%u reads=%u stuck=%u  %s\n",name,h[0],h[1],h[2],h[3],e?cudaGetErrorString(e):"");
    cudaFree(g);cudaFree(out);
}
int main(){ cudaFree(0);
    printf("hoist-proof; producer writes 1..%d\n",N);
    run<0,0>("GLOBAL relaxed.cta");
    run<1,0>("GLOBAL weak (LDG.E)");
    run<0,1>("SHARED relaxed.cta");
    run<1,1>("SHARED weak (LDS)");
    return 0;
}
