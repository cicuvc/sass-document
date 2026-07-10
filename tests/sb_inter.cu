// SB litmus — INTER-CTA control (actors in different CTAs => likely different SMs).
// Validates the harness: if store buffering exists anywhere, cross-SM SB(0,0)
// should appear. Contrast with the intra-CTA (intra-SM) result.
// Actor A = block (2k) tid0 ; Actor B = block (2k+1) tid0. Paired via global.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ void st_gpu(uint32_t*p,uint32_t v){asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ld_gpu(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ uint32_t ldrx(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}

// per-pair: X,Y (far apart) + arrival counter + result slots
struct Pair { uint32_t X; uint32_t pad0[63]; uint32_t Y; uint32_t pad1[63]; int arrive; uint32_t r0,r1; uint32_t pad2[61]; };

__global__ void sb_inter(Pair* pairs, unsigned long long* hist, int fenced){
    int pid = blockIdx.x>>1;      // pair id
    bool A = (blockIdx.x&1)==0;   // even block = actor A, odd = actor B
    if(threadIdx.x!=0) return;
    Pair* p=&pairs[pid];
    if(A){ p->X=0; p->r0=99; }
    if(!A){ p->Y=0; p->r1=99; }
    // arrival align across the two CTAs
    atomicAdd(&p->arrive,1);
    while(atomicAdd(&p->arrive,0)<2){}
    if(A){ st_gpu(&p->X,1); if(fenced)asm volatile("fence.gpu;":::"memory"); p->r0=ld_gpu(&p->Y); }
    else { st_gpu(&p->Y,1); if(fenced)asm volatile("fence.gpu;":::"memory"); p->r1=ld_gpu(&p->X); }
    // barrier so both wrote results before classify; reuse arrive
    atomicAdd(&p->arrive,1);
    while(atomicAdd(&p->arrive,0)<4){}
    if(A){ int idx=(p->r0!=0?1:0)*2+(p->r1!=0?1:0); atomicAdd(&hist[idx],1ULL); p->arrive=0; }
}

int main(){
    cudaFree(0);
    int npairs=2000, launches=3000;
    Pair* pairs; unsigned long long *hist,h[4];
    cudaMalloc(&pairs,sizeof(Pair)*npairs);
    cudaMalloc(&hist,sizeof(unsigned long long)*4);
    for(int fenced=0; fenced<2; ++fenced){
        cudaMemset(hist,0,sizeof(unsigned long long)*4);
        cudaMemset(pairs,0,sizeof(Pair)*npairs);
        for(int l=0;l<launches;l++) sb_inter<<<npairs*2,32>>>(pairs,hist,fenced);
        cudaError_t e=cudaDeviceSynchronize();
        cudaMemcpy(h,hist,sizeof(unsigned long long)*4,cudaMemcpyDeviceToHost);
        unsigned long long tot=h[0]+h[1]+h[2]+h[3];
        printf("inter-CTA .gpu %-9s tot=%llu  SB(0,0)=%llu  (0,1)=%llu (1,0)=%llu (1,1)=%llu  %s\n",
               fenced?"fence.gpu":"relaxed",tot,h[0],h[1],h[2],h[3], e?cudaGetErrorString(e):"");
    }
    return 0;
}
