#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Validate slice = parity of address bits {10,13,17,18,22,27,30,32,33}.
// Probe many random offsets from group-A SMs; predicted slice-0 should be fast,
// slice-1 slow. Report accuracy. Also retest bits 8-14 + pairs to resolve 11^12.
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
#define CHASE 2048
#define NP 48
__device__ uint32_t g_lat[512*NP]; __device__ unsigned g_sm[512];
__device__ __forceinline__ uint32_t chase1(char* p){
    uint64_t off=0,t0,t1; asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)(p+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1)); if(off==0xdead)*(volatile uint64_t*)p=off;
    return (uint32_t)((t1-t0)/CHASE);
}
__global__ void setup(char* buf, unsigned long long* offs, int n){ for(int i=0;i<n;i++)*(uint64_t*)(buf+offs[i])=0; }
__global__ void probe(char* buf, unsigned long long* offs, int n){
    if(threadIdx.x)return; unsigned sm=smid(); uint32_t lat[NP];
    for(int i=0;i<n;i++)lat[i]=chase1(buf+offs[i]);
    if(atomicCAS(&g_sm[sm],0xffffffffu,sm)==0xffffffffu) for(int i=0;i<n;i++)g_lat[sm*NP+i]=lat[i];
}
static int parity(unsigned long long a){ int bits[]={10,13,17,18,22,27,30,32,33}; int p=0; for(int b:bits)p^=(a>>b)&1; return p; }
int main(){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    unsigned long long offs[NP]; int pred[NP]; int n=0;
    srand(12345);
    for(int i=0;i<NP;i++){ unsigned long long a=((unsigned long long)rand()<<20 ^ ((unsigned long long)rand()<<3)) & ((1ULL<<35)-1); a&=~0x7ULL; offs[n]=a; pred[n]=parity(a); n++; }
    size_t bytes=(1ULL<<35)+(1<<20); char* buf; if(cudaMalloc(&buf,bytes)){printf("malloc fail\n");return 1;} cudaMemset(buf,0,bytes);
    unsigned long long* d; cudaMalloc(&d,n*8); cudaMemcpy(d,offs,n*8,cudaMemcpyHostToDevice);
    setup<<<1,1>>>(buf,d,n);
    unsigned fss[512]; for(int i=0;i<512;i++)fss[i]=0xffffffff; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss)); cudaDeviceSynchronize();
    probe<<<SMs*3,1>>>(buf,d,n); cudaDeviceSynchronize();
    uint32_t lat[512*NP]; unsigned sm[512]; cudaMemcpyFromSymbol(lat,g_lat,sizeof(lat)); cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    // group A = SMs with low latency on a known slice-0 addr (offset 0 -> parity 0)
    // use average latency over all probes to pick a detector group by first probe class
    // Simpler: pick SMs whose mean latency on predicted-slice0 set is low.
    double best=1e9; int det=-1;
    for(int s=0;s<SMs;s++){ if(sm[s]==0xffffffff)continue; double m=0;int c=0; for(int i=0;i<n;i++)if(pred[i]==0){m+=lat[s*NP+i];c++;} m/=c; if(m<best){best=m;det=s;} }
    // classify each probe by detector: near/far via threshold = midpoint
    double lo=1e9,hi=0; for(int i=0;i<n;i++){double v=lat[det*NP+i]; if(v<lo)lo=v; if(v>hi)hi=v;} double thr=(lo+hi)/2;
    int correct=0; printf("detector SM=%d near~%.0f far~%.0f thr~%.0f\n",det,lo,hi,thr);
    for(int i=0;i<n;i++){ int meas=lat[det*NP+i]<thr?0:1; if(meas==pred[i])correct++; }
    printf("linear-model accuracy: %d/%d = %.1f%%\n",correct,n,100.0*correct/n);
    // show mismatches
    for(int i=0;i<n;i++){ int meas=lat[det*NP+i]<thr?0:1; if(meas!=pred[i]) printf("  MISS off=0x%llx pred=%d meas=%d lat=%u\n",offs[i],pred[i],meas,lat[det*NP+i]); }
    return 0;
}
