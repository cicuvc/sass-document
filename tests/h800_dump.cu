#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Dump a labeled address->slice dataset from H800. One block/SM measures L2
// latency (ld.cg single-line chase) to N random addresses. Host splits SMs into
// two NUMA groups (by latency on a reference address), then labels each address
// by which group is faster (relative -> robust). Writes CSV: off_hex,latA,latB,label
#define CHASE 512
__device__ uint32_t* g_lat; __device__ unsigned g_sm[1024];
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
__device__ __forceinline__ uint32_t chase1(char* p){
    uint64_t off=0,t0,t1; asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)(p+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1)); if(off==0xdead)*(volatile uint64_t*)p=off; return (uint32_t)((t1-t0)/CHASE);
}
__global__ void setup(char* b, unsigned long long* o, int n){ for(long i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) *(uint64_t*)(b+o[i])=0; }
__global__ void probe(char* b, unsigned long long* o, int n, uint32_t* mat, int stride){
    if(threadIdx.x)return; unsigned sm=smid();
    // one row per SM (first block to claim the SM)
    if(atomicCAS(&g_sm[sm],0xffffffffu,sm)!=0xffffffffu) return;
    uint32_t* row=mat+(size_t)sm*stride;
    for(int i=0;i<n;i++) row[i]=chase1(b+o[i]);
}
int main(int argc,char**argv){
    int N = argc>1?atoi(argv[1]):32768;
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    unsigned long long MAXA=(1ULL<<34);   // 16 GB address space sampled
    printf("H800 SMs=%d N=%d span=%lluGB\n",SMs,N,MAXA>>30);
    unsigned long long* o=(unsigned long long*)malloc(N*8);
    srand(2024);
    for(int i=0;i<N;i++){ unsigned long long a=((unsigned long long)rand()<<20 ^ ((unsigned long long)rand()<<3) ^ rand()) & (MAXA-1); o[i]=a & ~0x7ULL; }
    char* buf; if(cudaMalloc(&buf,MAXA+ (1<<20))){printf("malloc fail\n");return 1;} cudaMemset(buf,0,MAXA+(1<<20));
    unsigned long long* d; cudaMalloc(&d,(size_t)N*8); cudaMemcpy(d,o,(size_t)N*8,cudaMemcpyHostToDevice);
    setup<<<256,256>>>(buf,d,N); cudaDeviceSynchronize();
    uint32_t* mat; int stride=N; cudaMalloc(&mat,(size_t)512*stride*4); cudaMemset(mat,0,(size_t)512*stride*4);
    cudaMemcpyToSymbol(g_lat,&mat,sizeof(mat));
    unsigned fss[1024]; for(int i=0;i<1024;i++)fss[i]=0xffffffff; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss));
    probe<<<SMs*4,1>>>(buf,d,N,mat,stride);
    cudaError_t e=cudaDeviceSynchronize();
    printf("probe err=%s\n",e?cudaGetErrorString(e):"ok");
    uint32_t* h=(uint32_t*)malloc((size_t)512*stride*4); cudaMemcpy(h,mat,(size_t)512*stride*4,cudaMemcpyDeviceToHost);
    unsigned sm[1024]; cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    // valid SMs
    int valid[512],nv=0; for(int s=0;s<SMs;s++) if(sm[s]!=0xffffffff) valid[nv++]=s;
    // group by reference address index 0: fast SMs -> G0, slow -> G1
    double ref=0; for(int k=0;k<nv;k++) ref+=h[(size_t)valid[k]*stride+0]; ref/=nv;
    int G0[512],G1[512],n0=0,n1=0;
    for(int k=0;k<nv;k++){ int s=valid[k]; (h[(size_t)s*stride+0]<ref? G0[n0++]:G1[n1++])=s; }
    printf("group0 n=%d group1 n=%d (ref addr slice0)\n",n0,n1);
    FILE* f=fopen("h800_slice_dataset.csv","w"); fprintf(f,"offset,latA,latB,label\n");
    int lab1=0;
    for(int i=0;i<N;i++){ double a=0,b=0; for(int k=0;k<n0;k++)a+=h[(size_t)G0[k]*stride+i]; a/=n0;
        for(int k=0;k<n1;k++)b+=h[(size_t)G1[k]*stride+i]; b/=n1;
        int label = a<b?0:1; lab1+=label;
        fprintf(f,"0x%llx,%.1f,%.1f,%d\n",o[i],a,b,label); }
    fclose(f);
    printf("wrote h800_slice_dataset.csv  N=%d  label1frac=%.3f\n",N,(double)lab1/N);
    return 0;
}
