#include <cstdint>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__device__ __forceinline__ void mark(volatile uint32_t*h,int i,uint32_t v){
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0],%1;"::"l"(h+i),"r"(v):"memory");
}
extern "C" __global__ void __cluster_dims__(2,1,1) victim(volatile uint32_t* h, uint32_t* good, int bad_rank){
    __shared__ uint32_t smem[64];
    unsigned rank = cg::this_cluster().block_rank();
    if(rank==0){
        uint32_t local=(uint32_t)__cvta_generic_to_shared(&smem[0]); uint32_t ds;
        asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(ds):"r"(local),"r"(bad_rank));
        asm volatile("st.shared::cluster.u32 [%0],%1;"::"r"(ds),"r"(0x1234):"memory");  // async cluster-fabric fault
        __threadfence();                                                                // MEMBAR ERRBAR CGAERRBAR CCTL
        asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(1):"memory");   // patch target
        mark(h,4,0xD09E);
    }
    cg::this_cluster().sync();
}
