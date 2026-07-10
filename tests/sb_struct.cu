// Structural SB probe. Each actor: store OWN cell, then load PEER cell (no
// same-address forwarding), then load OWN cell (forwarding/own-visibility check).
//   A0: X=1 ; rp0=Y(peer) ; rs0=X(own)
//   A1: Y=1 ; rp1=X(peer) ; rs1=Y(own)
// peer(0,0) = both peer-loads missed both stores = store buffering signature.
// self==0  = own store not yet visible to itself (would imply no forwarding AND
//            slow commit). Placement confirmed same-SM/diff-subpartition.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
__device__ __forceinline__ unsigned warpid(){unsigned r;asm volatile("mov.u32 %0,%%warpid;":"=r"(r));return r;}
__device__ __forceinline__ void stg(uint32_t*p,uint32_t v){asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ldg(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void sts(uint32_t a,uint32_t v){asm volatile("st.relaxed.cta.shared.u32 [%0],%1;"::"r"(a),"r"(v):"memory");}
__device__ __forceinline__ uint32_t lds(uint32_t a){uint32_t v;asm volatile("ld.relaxed.cta.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}

// hist[0..3]=peer outcome buckets, [4]=rejected placement, [5]=self-miss count, [6]=counted
template<int SPACE>
__global__ void probe(uint32_t* g, unsigned long long* hist){
    __shared__ uint32_t s[128]; __shared__ int arrive;
    __shared__ uint32_t rp0,rp1,rs0,rs1; __shared__ unsigned wa,wb,sa,sb;
    uint32_t* X=g+blockIdx.x*128+0; uint32_t* Y=g+blockIdx.x*128+64;
    uint32_t sX=(uint32_t)__cvta_generic_to_shared(&s[0]), sY=(uint32_t)__cvta_generic_to_shared(&s[64]);
    int tid=threadIdx.x, warp=tid>>5;
    bool a0=(warp==0&&(tid&31)==0), a1=(warp==1&&(tid&31)==0);
    if(tid==0){ if(SPACE){s[0]=0;s[64]=0;} else {X[0]=0;Y[0]=0;} arrive=0; }
    __syncthreads();
    if(a0){wa=warpid();sa=smid();} if(a1){wb=warpid();sb=smid();}
    if(a0||a1){
        atomicAdd(&arrive,1); while(atomicAdd(&arrive,0)<2){}
        if(a0){ if(SPACE){sts(sX,1); rp0=lds(sY); rs0=lds(sX);} else {stg(X,1); rp0=ldg(Y); rs0=ldg(X);} }
        else  { if(SPACE){sts(sY,1); rp1=lds(sX); rs1=lds(sY);} else {stg(Y,1); rp1=ldg(X); rs1=ldg(Y);} }
    }
    __syncthreads();
    if(tid==0){
        if(sa==sb && ((wa&3)!=(wb&3))){
            int idx=(rp0!=0?1:0)*2+(rp1!=0?1:0);
            atomicAdd(&hist[idx],1ULL); atomicAdd(&hist[6],1ULL);
            if(rs0==0||rs1==0) atomicAdd(&hist[5],1ULL);
        } else atomicAdd(&hist[4],1ULL);
    }
}
template<int SPACE> void run(const char* nm){
    int nblk=4096, launches=1500; uint32_t* g; unsigned long long *hist,h[7];
    cudaMalloc(&g,sizeof(uint32_t)*nblk*128); cudaMalloc(&hist,sizeof(unsigned long long)*7); cudaMemset(hist,0,sizeof(unsigned long long)*7);
    for(int l=0;l<launches;l++) probe<SPACE><<<nblk,64>>>(g,hist);
    cudaError_t e=cudaDeviceSynchronize(); cudaMemcpy(h,hist,sizeof(unsigned long long)*7,cudaMemcpyDeviceToHost);
    printf("%-8s counted=%llu | peer(0,0)=%llu (0,1)=%llu (1,0)=%llu (1,1)=%llu | self-miss=%llu rejected=%llu %s\n",
           nm,h[6],h[0],h[1],h[2],h[3],h[5],h[4], e?cudaGetErrorString(e):"");
    cudaFree(g); cudaFree(hist);
}
int main(){ cudaFree(0);
    printf("store-own then load-peer (forwarding excluded); + load-own (self check)\n");
    run<0>("GLOBAL"); run<1>("SHARED");
    return 0;
}
