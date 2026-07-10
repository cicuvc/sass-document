#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Direct GF(2) linearity test of the slice hash. For random a,b measure
// slice(a), slice(b), slice(a^b) via a detector SM; linear iff s(a^b)=s(a)^s(b).
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
#define CHASE 2048
#define NP 60   // 20 triples (a,b,a^b)
__device__ uint32_t g_lat[512*NP]; __device__ unsigned g_sm[512];
__device__ __forceinline__ uint32_t chase1(char* p){
    uint64_t off=0,t0,t1; asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)(p+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1)); if(off==0xdead)*(volatile uint64_t*)p=off; return (uint32_t)((t1-t0)/CHASE);
}
__global__ void setup(char* b, unsigned long long* o,int n){for(int i=0;i<n;i++)*(uint64_t*)(b+o[i])=0;}
__global__ void probe(char* b, unsigned long long* o,int n){ if(threadIdx.x)return; unsigned sm=smid(); uint32_t l[NP];
    for(int i=0;i<n;i++)l[i]=chase1(b+o[i]); if(atomicCAS(&g_sm[sm],~0u,sm)==~0u)for(int i=0;i<n;i++)g_lat[sm*NP+i]=l[i]; }
int main(){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    unsigned long long o[NP]; int T=NP/3; srand(7);
    auto R=[&](){ return (((unsigned long long)rand()<<20)^((unsigned long long)rand()<<3)) & ((1ULL<<34)-1) & ~0x7ULL; };
    for(int t=0;t<T;t++){ unsigned long long a=R(),b=R(); o[t*3]=a;o[t*3+1]=b;o[t*3+2]=a^b; }
    size_t bytes=(1ULL<<34)+(1<<20); char* buf; if(cudaMalloc(&buf,bytes)){printf("malloc fail\n");return 1;} cudaMemset(buf,0,bytes);
    unsigned long long* d; cudaMalloc(&d,NP*8); cudaMemcpy(d,o,NP*8,cudaMemcpyHostToDevice); setup<<<1,1>>>(buf,d,NP);
    unsigned fss[512]; for(int i=0;i<512;i++)fss[i]=~0u; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss)); cudaDeviceSynchronize();
    probe<<<SMs*3,1>>>(buf,d,NP); cudaDeviceSynchronize();
    uint32_t lat[512*NP]; unsigned sm[512]; cudaMemcpyFromSymbol(lat,g_lat,sizeof(lat)); cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    // detector = SM with widest spread (clear bimodal)
    int det=-1; double bestspread=0;
    for(int s=0;s<SMs;s++){ if(sm[s]==~0u)continue; double lo=1e9,hi=0; for(int i=0;i<NP;i++){double v=lat[s*NP+i]; if(v<lo)lo=v; if(v>hi)hi=v;} if(hi-lo>bestspread){bestspread=hi-lo;det=s;} }
    double lo=1e9,hi=0; for(int i=0;i<NP;i++){double v=lat[det*NP+i]; if(v<lo)lo=v; if(v>hi)hi=v;} double thr=(lo+hi)/2;
    printf("detector SM=%d near~%.0f far~%.0f\n",det,lo,hi);
    int lin=0; for(int t=0;t<T;t++){ int sa=lat[det*NP+t*3]<thr?0:1, sb=lat[det*NP+t*3+1]<thr?0:1, sab=lat[det*NP+t*3+2]<thr?0:1;
        int ok=(sab==(sa^sb)); lin+=ok; printf("t%-2d s(a)=%d s(b)=%d s(a^b)=%d  a^b_pred=%d  %s\n",t,sa,sb,sab,sa^sb,ok?"OK":"NONLINEAR"); }
    printf("linear triples: %d/%d\n",lin,T);
    return 0;
}
