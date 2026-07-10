// Coherence / staleness test across subpartitions within one CTA.
// Producer (warp0 tid0) writes an incrementing counter; Consumer (warp1 tid32)
// spin-reads. Questions:
//   - liveness: does the consumer see the counter advance to (near) the end?
//   - monotonicity: does it ever go backwards (incoherent stale copy)?
//   - does WEAK (non-scoped) load stay fresh, or only relaxed.cta?
// Private incoherent per-subcore copies (no .cta invalidation) => consumer STUCK.
// Single shared L1 / coherent structure => consumer tracks producer live.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define N 1000000

__device__ __forceinline__ void st_rlx(uint32_t*p,uint32_t v){asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ld_rlx(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ uint32_t ld_weak(uint32_t*p){uint32_t v;asm volatile("ld.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}

// mode: 0 = consumer relaxed.cta, 1 = consumer weak (plain LDG.E)
template<int WEAKLD, int SHARED>
__global__ void coh(uint32_t* g, uint32_t* out){
    __shared__ uint32_t sflag;
    uint32_t* flag = g;                    // one shared cell in L1
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(&sflag);
    int tid=threadIdx.x, warp=tid>>5;
    if(tid==0){ if(SHARED) sflag=0; else flag[0]=0; }
    __syncthreads();
    if(warp==0 && tid==0){
        // producer
        for(uint32_t j=1;j<=N;j++){ if(SHARED) asm volatile("st.relaxed.cta.shared.u32 [%0],%1;"::"r"(sa),"r"(j):"memory"); else st_rlx(flag,j); }
    } else if(warp==1 && tid==32){
        // consumer: track max seen, backward count, and last value
        uint32_t maxv=0, last=0; unsigned long long backward=0, reads=0; uint32_t stuck=0, prevmax=0, stuckmax=0;
        while(true){
            uint32_t v;
            if(SHARED){ if(WEAKLD) asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(sa):"memory"); else asm volatile("ld.relaxed.cta.shared.u32 %0,[%1];":"=r"(v):"r"(sa):"memory"); }
            else       v = WEAKLD? ld_weak(flag) : ld_rlx(flag);
            reads++;
            if(v<last) backward++;
            last=v; if(v>maxv) maxv=v;
            if(v>=N) break;
            // detect stuck: if maxv hasn't moved for a long time, bail
            if(maxv==prevmax){ if(++stuck>200000000u){ stuckmax=maxv; break; } }
            else { stuck=0; prevmax=maxv; }
        }
        out[0]=maxv; out[1]=(uint32_t)backward; out[2]=(uint32_t)reads; out[3]=stuckmax;
    }
}

template<int WEAKLD,int SHARED>
void run(const char* name){
    uint32_t *g,*out,h[4];
    cudaMalloc(&g,sizeof(uint32_t)*4); cudaMalloc(&out,sizeof(uint32_t)*4);
    cudaMemset(out,0,sizeof(uint32_t)*4);
    coh<WEAKLD,SHARED><<<1,64>>>(g,out);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(h,out,sizeof(uint32_t)*4,cudaMemcpyDeviceToHost);
    printf("%-28s maxSeen=%u backward=%u reads=%u stuckAt=%u  %s\n",
           name,h[0],h[1],h[2],h[3], e?cudaGetErrorString(e):"");
    cudaFree(g);cudaFree(out);
}
int main(){
    cudaFree(0);
    printf("producer writes 1..%d ; consumer on a different subpartition\n",N);
    run<0,0>("GLOBAL consumer relaxed.cta");
    run<1,0>("GLOBAL consumer WEAK (LDG.E)");
    run<0,1>("SHARED consumer relaxed.cta");
    run<1,1>("SHARED consumer WEAK (LDS)");
    return 0;
}
