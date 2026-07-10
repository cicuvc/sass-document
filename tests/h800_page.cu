#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
// Dump COMPLETE within-2MB-page slice map. Only within-page bits (7..20) vary,
// so VA offset == PA offset -> hash should be linear/recoverable. All 16384
// cache lines (128B) in one 2MB-aligned page.
#define CHASE 512
#define NLINE 16384    // 2MB / 128B
__device__ unsigned g_sm[1024];
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
__device__ __forceinline__ uint32_t chase1(char* p){ uint64_t off=0,t0,t1;
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)(p+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1)); if(off==0xdead)*(volatile uint64_t*)p=off; return (uint32_t)((t1-t0)/CHASE); }
__global__ void setup(char* b){ for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<NLINE;i+=gridDim.x*blockDim.x)*(uint64_t*)(b+(size_t)i*128)=0; }
__global__ void probe(char* b, uint32_t* mat, int stride){
    if(threadIdx.x)return; unsigned sm=smid(); if(atomicCAS(&g_sm[sm],0xffffffffu,sm)!=0xffffffffu)return;
    uint32_t* row=mat+(size_t)sm*stride; for(int i=0;i<NLINE;i++) row[i]=chase1(b+(size_t)i*128);
}
int main(){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    char* buf; cudaMalloc(&buf,(2<<20)+4096);  // one 2MB page (cudaMalloc is 2MB-aligned)
    cudaMemset(buf,0,(2<<20)); setup<<<64,256>>>(buf); cudaDeviceSynchronize();
    uint32_t* mat; int stride=NLINE; cudaMalloc(&mat,(size_t)512*stride*4); cudaMemset(mat,0,(size_t)512*stride*4);
    unsigned fss[1024]; for(int i=0;i<1024;i++)fss[i]=0xffffffff; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss));
    probe<<<SMs*4,1>>>(buf,mat,stride); cudaError_t e=cudaDeviceSynchronize();
    printf("err=%s SMs=%d\n",e?cudaGetErrorString(e):"ok",SMs);
    uint32_t* h=(uint32_t*)malloc((size_t)512*stride*4); cudaMemcpy(h,mat,(size_t)512*stride*4,cudaMemcpyDeviceToHost);
    unsigned sm[1024]; cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    int valid[512],nv=0; for(int s=0;s<SMs;s++)if(sm[s]!=0xffffffff)valid[nv++]=s;
    double ref=0; for(int k=0;k<nv;k++)ref+=h[(size_t)valid[k]*stride+0]; ref/=nv;
    int G0[512],G1[512],n0=0,n1=0; for(int k=0;k<nv;k++){int s=valid[k];(h[(size_t)s*stride+0]<ref?G0[n0++]:G1[n1++])=s;}
    FILE* f=fopen("h800_page_map.csv","w"); fprintf(f,"line,offset,latA,latB,label\n"); int l1=0;
    for(int i=0;i<NLINE;i++){ double a=0,b=0; for(int k=0;k<n0;k++)a+=h[(size_t)G0[k]*stride+i]; a/=n0;
        for(int k=0;k<n1;k++)b+=h[(size_t)G1[k]*stride+i]; b/=n1; int lab=a<b?0:1; l1+=lab;
        fprintf(f,"%d,0x%x,%.1f,%.1f,%d\n",i,i*128,a,b,lab); }
    fclose(f); printf("wrote h800_page_map.csv NLINE=%d label1frac=%.3f g0=%d g1=%d\n",NLINE,(double)l1/NLINE,n0,n1);
    return 0;
}
