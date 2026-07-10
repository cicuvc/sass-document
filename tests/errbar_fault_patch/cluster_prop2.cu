#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
// No cluster.sync. rank1 RO-faults (async) then sets a global flag. rank0 waits
// for the flag (so peer's fault is pending), then runs a long loop of markers,
// with (fencemode=1) or without (fencemode=0) __threadfence()/CGAERRBAR each
// iter. If CGAERRBAR propagates the peer's cluster error, the fence version's
// markers should stop EARLIER than the no-fence version. If both stop at the
// same point, it is generic grid teardown, not CGAERRBAR.
__device__ __forceinline__ void mark(volatile uint32_t* h,int i,uint32_t v){
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0],%1;"::"l"(h+i),"r"(v):"memory");
}
extern "C" __global__ void __cluster_dims__(2,1,1)
probe(volatile uint32_t* h, uint32_t* bad, uint32_t* good, uint32_t* flag, int mode, int fencemode){
    unsigned rank = cg::this_cluster().block_rank();
    if(rank==1){
        uint32_t* p = mode? bad : good;
        asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(p),"r"(1):"memory"); // async fault
        asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(flag),"r"(1):"memory"); // signal issued
        mark(h,20,1);
        // rank1 lingers so its fault surfaces late (at exit), not immediately
        for(int i=0;i<100000;i++) asm volatile("":::"memory");
        mark(h,21,1);
    } else {
        // rank0: wait until peer signaled
        uint32_t f=0; int g=0;
        do { asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(f):"l"(flag):"memory"); } while(f==0 && ++g<(1<<24));
        mark(h,0,1);                    // saw peer's signal
        for(int i=1;i<=200;i++){
            if(fencemode) __threadfence();   // CGAERRBAR each iter
            mark(h,1,i);                     // furthest progress
        }
        mark(h,4,0xD09E);
    }
}
static CUdeviceptr ro_region(){
    cuInit(0); CUcontext ctx; cuCtxGetCurrent(&ctx);
    if(!ctx){ CUdevice d; cuDeviceGet(&d,0); cuCtxCreate(&ctx,NULL,0,d); }
    CUmemAllocationProp prop={}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=0;
    size_t gran; cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    CUmemGenericAllocationHandle h; cuMemCreate(&h,gran,&prop,0);
    CUdeviceptr p; cuMemAddressReserve(&p,gran,0,0,0); cuMemMap(p,gran,0,h,0);
    CUmemAccessDesc ad={}; ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id=0;
    ad.flags=CU_MEM_ACCESS_FLAGS_PROT_READ; cuMemSetAccess(p,gran,&ad,1); return p;
}
int main(int argc,char**argv){
    int mode=argc>1?atoi(argv[1]):1, fencemode=argc>2?atoi(argv[2]):1;
    uint32_t* hbuf; cudaHostAlloc(&hbuf,256,cudaHostAllocMapped); for(int i=0;i<64;i++)hbuf[i]=0;
    volatile uint32_t* dh; cudaHostGetDevicePointer((void**)&dh,hbuf,0);
    uint32_t* good; cudaMalloc(&good,4096); cudaMemset(good,0,4096);
    uint32_t* flag; cudaMalloc(&flag,4); cudaMemset(flag,0,4);
    uint32_t* bad = mode? (uint32_t*)ro_region() : good;
    probe<<<2,1>>>(dh,bad,good,flag,mode,fencemode);
    cudaError_t e=cudaDeviceSynchronize();
    printf("mode=%d fence=%d sync=%s | obs saw[0]=%u furthest[1]=%u DONE[4]=0x%x | peer stored[20]=%u end[21]=%u\n",
        mode,fencemode,cudaGetErrorString(e),hbuf[0],hbuf[1],hbuf[4],hbuf[20],hbuf[21]);
    return 0;
}
