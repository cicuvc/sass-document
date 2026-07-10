#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__device__ __forceinline__ void mark(volatile uint32_t*h,int i,uint32_t v){
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0],%1;"::"l"(h+i),"r"(v):"memory");
}
// store to a DSMEM address in a (possibly invalid) peer rank
extern "C" __global__ void __cluster_dims__(2,1,1) victim(volatile uint32_t* h, int bad_rank){
    __shared__ uint32_t smem[64];
    unsigned rank = cg::this_cluster().block_rank();
    if(rank==0){
        uint32_t local=(uint32_t)__cvta_generic_to_shared(&smem[0]);
        uint32_t ds;
        asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(ds):"r"(local),"r"(bad_rank));
        mark(h,0,1);                    // before dsmem store
        asm volatile("st.shared::cluster.u32 [%0],%1;"::"r"(ds),"r"(0x1234):"memory");
        mark(h,1,1);                    // after dsmem store (present => async)
        __threadfence();
        mark(h,2,1);                    // after fence
    }
    cg::this_cluster().sync();
}
int main(int argc,char**argv){
    int bad_rank=argc>1?atoi(argv[1]):7;   // 7 = invalid in a 2-CTA cluster
    uint32_t* hbuf; cudaHostAlloc(&hbuf,256,cudaHostAllocMapped); for(int i=0;i<64;i++)hbuf[i]=0;
    volatile uint32_t* dh; cudaHostGetDevicePointer((void**)&dh,hbuf,0);
    victim<<<2,1>>>(dh,bad_rank);
    cudaError_t e=cudaDeviceSynchronize();
    printf("bad_rank=%d sync=%s | before[0]=%u after[1]=%u fence[2]=%u\n",
        bad_rank,cudaGetErrorString(e),hbuf[0],hbuf[1],hbuf[2]);
    return 0;
}
