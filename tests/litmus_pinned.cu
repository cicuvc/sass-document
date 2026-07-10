// Hardened intra-SM litmus with sub-partition pinning + MP positive control.
//   %smid  -> which SM   ;  %warpid -> physical warp slot ; subpart = warpid % 4
// Only samples where the two actors are (same SM) && (different sub-partition)
// are counted. MP is a positive control: relaxed-no-fence weak reordering should
// be observable (proving the harness detects intra-SM weak behaviour), while SB
// (0,0) should stay absent.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
__device__ __forceinline__ unsigned warpid(){unsigned r;asm volatile("mov.u32 %0,%%warpid;":"=r"(r));return r;}
__device__ __forceinline__ void stg(uint32_t*p,uint32_t v){asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ uint32_t ldg(uint32_t*p){uint32_t v;asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void fen(int m){if(m==1)asm volatile("fence.cta;":::"memory");else if(m==2)asm volatile("fence.sc.cta;":::"memory");}

// hist[5]: 0..3 = SB/MP outcome buckets, 4 = #samples rejected (placement bad)
// SB: A0: X=1;fence;r0=Y   A1: Y=1;fence;r1=X   bucket=(r0!=0)*2+(r1!=0), (0,0)=bucket0
// MP: P : D=1;fence;F=1    C : rf=F;fence;rd=D  weak(bad)= rf==1&&rd==0 -> bucket tracked
template<int FENCE, int MP>
__global__ void litmus(uint32_t* g, unsigned long long* hist, unsigned* place){
    __shared__ int arrive; __shared__ uint32_t rA,rB; __shared__ unsigned wa,wb,sa_,sb_;
    uint32_t* X=g+blockIdx.x*128+0; uint32_t* Y=g+blockIdx.x*128+64;
    int tid=threadIdx.x, warp=tid>>5;
    bool a0=(warp==0&&(tid&31)==0), a1=(warp==1&&(tid&31)==0);
    if(tid==0){ X[0]=0; Y[0]=0; arrive=0; rA=99; rB=99; }
    __syncthreads();
    if(a0){ wa=warpid(); sa_=smid(); }
    if(a1){ wb=warpid(); sb_=smid(); }
    if(a0||a1){
        atomicAdd(&arrive,1);
        while(atomicAdd(&arrive,0)<2){}
        if(!MP){ // Store Buffering
            if(a0){ stg(X,1); fen(FENCE); rA=ldg(Y); }
            else  { stg(Y,1); fen(FENCE); rB=ldg(X); }
        } else {  // Message Passing: a0=producer (X=data,Y=flag), a1=consumer
            if(a0){ stg(X,1); fen(FENCE); stg(Y,1); }
            else  { rB=ldg(Y); fen(FENCE); rA=ldg(X); }  // rB=flag, rA=data
        }
    }
    __syncthreads();
    if(tid==0){
        bool sameSM = (sa_==sb_);
        bool diffSub = ((wa&3)!=(wb&3));
        if(sameSM && diffSub){
            if(!MP){ int idx=(rA!=0?1:0)*2+(rB!=0?1:0); atomicAdd(&hist[idx],1ULL); }
            else { // consumer: rB=flag, rA=data. weak bad outcome = flag==1 && data==0
                int idx = (rB!=0 && rA==0)?0 : (rB!=0 && rA!=0)?1 : (rB==0)?2 : 3;
                atomicAdd(&hist[idx],1ULL);
            }
            atomicAdd(&place[0],1u);
        } else atomicAdd(&hist[4],1ULL);
        // record example placement
        if(blockIdx.x==0){ place[1]=sa_; place[2]=sb_; place[3]=wa; place[4]=wb; }
    }
}

template<int FENCE,int MP> void run(const char* name){
    int nblk=4096, launches=1500;
    uint32_t* g; unsigned long long *hist,h[5]; unsigned *place,pl[5];
    cudaMalloc(&g,sizeof(uint32_t)*nblk*128);
    cudaMalloc(&hist,sizeof(unsigned long long)*5); cudaMemset(hist,0,sizeof(unsigned long long)*5);
    cudaMalloc(&place,sizeof(unsigned)*5); cudaMemset(place,0,sizeof(unsigned)*5);
    for(int l=0;l<launches;l++) litmus<FENCE,MP><<<nblk,64>>>(g,hist,place);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(h,hist,sizeof(unsigned long long)*5,cudaMemcpyDeviceToHost);
    cudaMemcpy(pl,place,sizeof(unsigned)*5,cudaMemcpyDeviceToHost);
    unsigned long long ok=h[0]+h[1]+h[2]+h[3];
    if(!MP) printf("SB  %-14s counted=%llu rejected=%llu | SB(0,0)=%llu (0,1)=%llu (1,0)=%llu (1,1)=%llu | ex sm(%u,%u) warp(%u,%u) %s\n",
        name,ok,h[4],h[0],h[1],h[2],h[3],pl[1],pl[2],pl[3],pl[4],e?cudaGetErrorString(e):"");
    else printf("MP  %-14s counted=%llu rejected=%llu | WEAK(flag1,data0)=%llu (1,1)=%llu (flag0)=%llu | ex sm(%u,%u) warp(%u,%u) %s\n",
        name,ok,h[4],h[0],h[1],h[2],pl[1],pl[2],pl[3],pl[4],e?cudaGetErrorString(e):"");
    cudaFree(g);cudaFree(hist);cudaFree(place);
}
int main(){ cudaFree(0);
    printf("== Store Buffering (only same-SM, different-subpartition samples) ==\n");
    run<0,0>("relaxed"); run<1,0>("fence.cta"); run<2,0>("fence.sc");
    printf("== Message Passing positive control ==\n");
    run<0,1>("relaxed"); run<1,1>("fence.cta"); run<2,1>("fence.sc");
    return 0;
}
