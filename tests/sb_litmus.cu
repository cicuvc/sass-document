// SB litmus v3 — tightly aligned actors to sample the concurrent interleaving.
// Two actors spin on a shared arrival counter and are released together, then
// immediately do store; (fence); load. One test per CTA, many CTAs, many launches.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ void stg(uint32_t*p,uint32_t v){asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ldg(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void sts(uint32_t a,uint32_t v){asm volatile("st.relaxed.cta.shared.u32 [%0],%1;"::"r"(a),"r"(v):"memory");}
__device__ __forceinline__ uint32_t lds(uint32_t a){uint32_t v;asm volatile("ld.relaxed.cta.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}
__device__ __forceinline__ void fence(int m){if(m==1)asm volatile("fence.cta;":::"memory");else if(m==2)asm volatile("fence.sc.cta;":::"memory");}

// X far from Y (distinct lines). Actors: warp0 tid0, warp1 tid32.
template<int SPACE,int FENCE>
__global__ void sb(uint32_t* g, unsigned long long* hist){
    __shared__ uint32_t s[128];
    __shared__ int arrive;
    __shared__ uint32_t rr0, rr1;
    uint32_t* X = g + blockIdx.x*128 + 0;
    uint32_t* Y = g + blockIdx.x*128 + 64;
    uint32_t sX=(uint32_t)__cvta_generic_to_shared(&s[0]);
    uint32_t sY=(uint32_t)__cvta_generic_to_shared(&s[64]);
    int tid=threadIdx.x, warp=tid>>5;
    bool a0=(warp==0&&(tid&31)==0), a1=(warp==1&&(tid&31)==0);
    if(tid==0){ if(SPACE){s[0]=0;s[64]=0;} else {X[0]=0;Y[0]=0;} arrive=0; rr0=99; rr1=99; }
    __syncthreads();
    if(a0||a1){
        atomicAdd(&arrive,1);
        while(atomicAdd(&arrive,0)<2){}          // release both ~together
        if(a0){ if(SPACE){sts(sX,1);fence(FENCE);rr0=lds(sY);} else {stg(X,1);fence(FENCE);rr0=ldg(Y);} }
        else  { if(SPACE){sts(sY,1);fence(FENCE);rr1=lds(sX);} else {stg(Y,1);fence(FENCE);rr1=ldg(X);} }
    }
    __syncthreads();
    if(tid==0){ int idx=(rr0!=0?1:0)*2+(rr1!=0?1:0); atomicAdd(&hist[idx],1ULL); }
}

template<int SPACE,int FENCE>
void run(const char* name){
    int nblk=4096, nthr=64, launches=2000;
    uint32_t* g; unsigned long long *hist,h[4];
    cudaMalloc(&g,sizeof(uint32_t)*nblk*128);
    cudaMalloc(&hist,sizeof(unsigned long long)*4);
    cudaMemset(hist,0,sizeof(unsigned long long)*4);
    for(int l=0;l<launches;l++) sb<SPACE,FENCE><<<nblk,nthr>>>(g,hist);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(h,hist,sizeof(unsigned long long)*4,cudaMemcpyDeviceToHost);
    unsigned long long tot=h[0]+h[1]+h[2]+h[3];
    printf("%-20s tot=%llu  SB(0,0)=%llu  (0,1)=%llu (1,0)=%llu (1,1)=%llu  %s\n",
           name,tot,h[0],h[1],h[2],h[3], e?cudaGetErrorString(e):"");
    cudaFree(g);cudaFree(hist);
}
int main(){
    cudaFree(0);
    printf("== GLOBAL .cta relaxed (LDG/STG.STRONG.SM) ==\n");
    run<0,0>("global relaxed"); run<0,1>("global fence.cta"); run<0,2>("global fence.sc");
    printf("== SHARED (LDS/STS) ==\n");
    run<1,0>("shared relaxed"); run<1,1>("shared fence.cta"); run<1,2>("shared fence.sc");
    return 0;
}
