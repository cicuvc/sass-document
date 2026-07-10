// MP litmus v4 — DATA DEPENDENCY that ptxas CANNOT constant-fold.
// Fix: D[f & 1] look-up. Consumer does d = ld(D_base[f & 1]) where D_base[]
// has D placed so index 0=triggering, index 1=triggering too (both same row,
// dep genuinely gates load order). Flag is 0→1 only, so f==1 at exit; index=1.
// ptxas cannot know f at compile time, so the address is genuinely computed.
// Intra: shared padded (D=s[0], F=s[64]); inter: D[2], F at offset 256.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
__device__ __forceinline__ void st_gpu(uint32_t*p,uint32_t v){asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ld_gpu(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void st_shared(uint32_t a,uint32_t v){asm volatile("st.relaxed.cta.shared.u32 [%0],%1;"::"r"(a),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ld_shared(uint32_t a){uint32_t v;asm volatile("ld.relaxed.cta.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}

// Inter: D_len is a volatile kernel arg so ptxas cannot fold the index.
__global__ void mp_inter_vol(uint32_t* pair, unsigned long long* hist, int fenced, volatile int D_len){
    int pid=blockIdx.x>>1; bool P=(blockIdx.x&1)==0; if(threadIdx.x!=0)return;
    unsigned* counters=(unsigned*)(pair + pid*134 + 3); uint32_t* D=((uint32_t*)counters)+4; uint32_t* F=D+64; unsigned* smP=counters+0; unsigned* smC=counters+1;
    if(P){ *smP=smid(); st_gpu(D+0,1); st_gpu(D+1,1); if(fenced)asm volatile("fence.gpu;":::"memory"); st_gpu(F,1); }
    else { *smC=smid(); uint32_t f=0; int g=0;
           while(f==0){ f=ld_gpu(F); if(++g>(1<<24))return; }
           // Real dep: ptxas doesn't know f, so D[f & 1] is a true computed address.
           // D_len volatile arg stops ptxas from bounding the index.
           uint32_t d=ld_gpu(D + (f & 1 & (D_len>>31))); // D_len>0 => mask=~0 => f&1 ; f=1 => idx=1
           atomicAdd(hist+0,1u); if(d==0) atomicAdd(hist+1,1u);
           if(*smP != *smC) atomicAdd(hist+2,1u); }
}

// Intra: shared padded, lookup idx = f & 1
__global__ void mp_intra_vol(uint32_t* g, unsigned* res){
    __shared__ uint32_t s[128]; int tid=threadIdx.x, warp=tid>>5;
    if(tid==0){ s[0]=0; s[4]=0; s[64]=0; } __syncthreads();
    uint32_t aD0=(uint32_t)__cvta_generic_to_shared(&s[0]);
    uint32_t aD1=(uint32_t)__cvta_generic_to_shared(&s[4]);
    uint32_t aF =(uint32_t)__cvta_generic_to_shared(&s[64]);
    if(warp==0 && tid==0){
        st_shared(aD0,1); st_shared(aD1,1);  // both D slots = 1
        st_shared(aF,1);
        unsigned w;asm volatile("mov.u32 %0,%%warpid;":"=r"(w));res[2]=w; res[4]=smid();
    } else if(warp==1 && tid==32){
        uint32_t f; int g=0;
        do { f=ld_shared(aF); } while(f==0 && ++g<(1<<24));
        // Real dep: (f & 1) ? aD1 : aD0.  f==1 → idx=1 → aD1 = s[1] = 1.
        uint32_t* addr = (f & 1) ? (uint32_t*)aD1 : (uint32_t*)aD0;
        uint32_t d; asm volatile("ld.relaxed.cta.shared.u32 %0,[%1];":"=r"(d):"r"((uint32_t)addr):"memory");
        atomicAdd(&res[0],1u); if(f!=0&&d==0) atomicAdd(&res[1],1u); unsigned w3;asm volatile("mov.u32 %0,%%warpid;":"=r"(w3));res[3]=w3; res[5]=smid();
    }
}

int main(){ cudaFree(0);
    // inter-SM
    int npairs=160, launches=60000; uint32_t* pair; unsigned long long *hist, hh[3];
    cudaMalloc(&pair,sizeof(uint32_t)*npairs*256); cudaMalloc(&hist,sizeof(unsigned long long)*3);
    int dummy_len=99;
    for(int fenced=0;fenced<2;++fenced){ cudaMemset(hist,0,sizeof(unsigned long long)*3); cudaMemset(pair,0,sizeof(uint32_t)*npairs*256);
        for(int l=0;l<launches;l++) mp_inter_vol<<<npairs*2,32>>>(pair,hist,fenced,dummy_len);
        cudaDeviceSynchronize(); cudaMemcpy(hh,hist,sizeof(unsigned long long)*3,cudaMemcpyDeviceToHost);
        printf("INTER .gpu %-7s seen=%llu WEAK=%llu diffSM=%llu\n",fenced?"fenced":"relaxed",hh[0],hh[1],hh[2]); }
    // intra-SM
    unsigned *res; cudaMalloc(&res,sizeof(unsigned)*10); unsigned hr[10];
    unsigned long long tot=0,weak=0;
    for(int l=0;l<200000;l++){ cudaMemset(res,0,sizeof(unsigned)*10); mp_intra_vol<<<1,64>>>(nullptr,res);
        cudaMemcpy(hr,res,sizeof(unsigned)*10,cudaMemcpyDeviceToHost); tot+=hr[0]; weak+=hr[1]; }
    cudaDeviceSynchronize();
    printf("INTRA .cta relaxed    samples=%llu WEAK(data==0)=%llu\n",tot,weak);
    return 0;
}
