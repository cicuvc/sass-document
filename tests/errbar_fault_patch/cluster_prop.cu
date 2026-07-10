#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// Cluster of 2 CTAs. rank1 = peer (stores to bad RO addr => async fault, or good).
// rank0 = observer: logs progress markers to pinned host mem (mmio.sys, survives
// fault) around cluster.sync and __threadfence (which contains CGAERRBAR).
// If rank1's fault propagates to rank0's cluster barrier / CGAERRBAR, rank0's
// markers stop early. Compare mode=0 (peer good) vs mode=1 (peer bad).

__device__ __forceinline__ void mark(volatile uint32_t* h,int i,uint32_t v){
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0],%1;"::"l"(h+i),"r"(v):"memory");
}
extern "C" __global__ void __cluster_dims__(2,1,1)
probe(volatile uint32_t* h, uint32_t* bad, uint32_t* good, int mode){
    cg::cluster_group cl = cg::this_cluster();
    unsigned rank = cl.block_rank();
    if(rank==0) mark(h,0,1);
    if(rank==1){
        uint32_t* p = mode? bad : good;
        asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(p),"r"(1):"memory");
        mark(h,20,1);
    }
    cl.sync();                          // both reach barrier
    if(rank==0){
        mark(h,1,1);                    // survived cluster.sync
        // rank1 has done its store; its fault (if async) is pending, but cluster.sync
        // already drained rank1's side. Let rank0 hit CGAERRBAR separately via fence.
        __threadfence();                // MEMBAR ERRBAR CGAERRBAR CCTL
        mark(h,2,1);                    // survived fence (CGAERRBAR)
        mark(h,4,0xD09E);
    }
    if(rank==1) mark(h,21,1);
}
int main(int argc,char**argv){
    int mode = argc>1?atoi(argv[1]):1;
    uint32_t* hbuf; cudaHostAlloc(&hbuf,256,cudaHostAllocMapped);
    for(int i=0;i<64;i++) hbuf[i]=0;
    volatile uint32_t* dh; cudaHostGetDevicePointer((void**)&dh,hbuf,0);
    uint32_t* good; cudaMalloc(&good,4096);
    // Allocate RO region for async fault
    CUdeviceptr ro_bad = 0;
    if(mode>=1){
        cuInit(0); CUdevice dev; CUcontext ctx; cuCtxGetCurrent(&ctx);
        if(!ctx){ cuDeviceGet(&dev,0); cuCtxCreate(&ctx,NULL,0,dev); }
        CUmemAllocationProp prop={}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=0;
        size_t gran; cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        CUmemGenericAllocationHandle h; cuMemCreate(&h,gran,&prop,0);
        cuMemAddressReserve(&ro_bad,gran,0,0,0); cuMemMap(ro_bad,gran,0,h,0);
        CUmemAccessDesc ad={}; ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id=0;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        cuMemSetAccess(ro_bad,gran,&ad,1);
        printf("RO region at 0x%llx\n",(unsigned long long)ro_bad);
    }
    uint32_t* bad = (mode==0)?good : (uint32_t*)ro_bad;    // mode0=good, mode1=RO async
    probe<<<2,1>>>(dh,bad,good,mode);
    cudaError_t e=cudaDeviceSynchronize();
    printf("mode=%d sync=%s\n  obs: start[0]=%u sync[1]=%u fence1[2]=%u loop[3]=%u DONE[4]=0x%x\n  peer: stored[20]=%u end[21]=%u\n",
        mode,cudaGetErrorString(e),hbuf[0],hbuf[1],hbuf[2],hbuf[3],hbuf[4],hbuf[20],hbuf[21]);
    return 0;
}
