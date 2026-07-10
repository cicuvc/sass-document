#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
// Physically-contiguous 2MB VMM granule -> within it, offset bits 7..20 == PA
// bits, so slice(offset) is linear/learnable. Dump the 16384-line within-granule
// slice map. (cuMemCreate 2MB = one large physical page => contiguous.)
#define CHASE 512
#define NLINE 16384
__device__ unsigned g_sm[1024];
__device__ __forceinline__ unsigned smid(){unsigned r;asm volatile("mov.u32 %0,%%smid;":"=r"(r));return r;}
__device__ __forceinline__ uint32_t chase1(char* p){ uint64_t off=0,t0,t1;
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t0));
    #pragma unroll 1
    for(int i=0;i<CHASE;i++) asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(off):"l"((uint64_t*)(p+off)):"memory");
    asm volatile("mov.u64 %0,%%clock64;":"=l"(t1)); if(off==0xdead)*(volatile uint64_t*)p=off; return (uint32_t)((t1-t0)/CHASE); }
__global__ void setup(char* b){ for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<NLINE;i+=gridDim.x*blockDim.x)*(uint64_t*)(b+(size_t)i*128)=0; }
__global__ void probe(char* b, uint32_t* mat){ if(threadIdx.x)return; unsigned sm=smid();
    if(atomicCAS(&g_sm[sm],0xffffffffu,sm)!=0xffffffffu)return; uint32_t* row=mat+(size_t)sm*NLINE;
    for(int i=0;i<NLINE;i++) row[i]=chase1(b+(size_t)i*128); }
int main(){
    cuInit(0); CUdevice dev; cuDeviceGet(&dev,0); CUcontext ctx; cuCtxCreate(&ctx,0,dev);
    CUmemAllocationProp prop={}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=dev;
    size_t gran; cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t sz=2ULL<<20; if(sz<gran)sz=gran;
    CUmemGenericAllocationHandle h; cuMemCreate(&h,sz,&prop,0);
    CUdeviceptr p; cuMemAddressReserve(&p,sz,0,0,0); cuMemMap(p,sz,0,h,0);
    CUmemAccessDesc ad={}; ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id=dev; ad.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(p,sz,&ad,1);
    printf("VMM granule=%zuKB mapped sz=%zuMB VA=0x%llx\n",gran>>10,sz>>20,(unsigned long long)p);
    char* buf=(char*)p; cudaMemset(buf,0,sz);
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); int SMs=pr.multiProcessorCount;
    setup<<<64,256>>>(buf); cudaDeviceSynchronize();
    uint32_t* mat; cudaMalloc(&mat,(size_t)512*NLINE*4); cudaMemset(mat,0,(size_t)512*NLINE*4);
    unsigned fss[1024]; for(int i=0;i<1024;i++)fss[i]=0xffffffff; cudaMemcpyToSymbol(g_sm,fss,sizeof(fss));
    probe<<<SMs*4,1>>>(buf,mat); cudaError_t e=cudaDeviceSynchronize(); printf("err=%s\n",e?cudaGetErrorString(e):"ok");
    uint32_t* hh=(uint32_t*)malloc((size_t)512*NLINE*4); cudaMemcpy(hh,mat,(size_t)512*NLINE*4,cudaMemcpyDeviceToHost);
    unsigned sm[1024]; cudaMemcpyFromSymbol(sm,g_sm,sizeof(sm));
    int valid[512],nv=0; for(int s=0;s<SMs;s++)if(sm[s]!=0xffffffff)valid[nv++]=s;
    double ref=0; for(int k=0;k<nv;k++)ref+=hh[(size_t)valid[k]*NLINE+0]; ref/=nv;
    int G0[512],G1[512],n0=0,n1=0; for(int k=0;k<nv;k++){int s=valid[k];(hh[(size_t)s*NLINE+0]<ref?G0[n0++]:G1[n1++])=s;}
    FILE* f=fopen("h800_vmm_map.csv","w"); fprintf(f,"offset,latA,latB,label\n"); int l1=0;
    for(int i=0;i<NLINE;i++){ double a=0,b=0; for(int k=0;k<n0;k++)a+=hh[(size_t)G0[k]*NLINE+i]; a/=n0;
        for(int k=0;k<n1;k++)b+=hh[(size_t)G1[k]*NLINE+i]; b/=n1; int lab=a<b?0:1; l1+=lab;
        fprintf(f,"0x%x,%.1f,%.1f,%d\n",i*128,a,b,lab);} fclose(f);
    printf("wrote h800_vmm_map.csv label1frac=%.3f\n",(double)l1/NLINE);
    return 0;
}
