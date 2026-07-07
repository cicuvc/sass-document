#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// local mbarrier, cluster-scope arrive (token form)
__global__ void __cluster_dims__(2,1,1) k_local(int *out){
    __shared__ unsigned long long bar;
    unsigned s = (unsigned)__cvta_generic_to_shared(&bar);
    if(threadIdx.x==0) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(s),"r"(blockDim.x));
    asm volatile("barrier.cluster.arrive; barrier.cluster.wait;");
    unsigned long long tok;
    asm volatile("mbarrier.arrive.release.cluster.shared::cta.b64 %0,[%1];":"=l"(tok):"r"(s));
    out[threadIdx.x]=(int)tok;
}
// remote mbarrier via mapa (distributed shared memory) — no token (remote)
__global__ void __cluster_dims__(2,1,1) k_remote(int *out){
    cg::cluster_group cl=cg::this_cluster();
    __shared__ unsigned long long bar;
    unsigned s=(unsigned)__cvta_generic_to_shared(&bar);
    if(threadIdx.x==0) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(s),"r"(blockDim.x));
    asm volatile("barrier.cluster.arrive; barrier.cluster.wait;");
    unsigned rank = cl.block_rank()^1, remote;
    asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(s),"r"(rank));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote));
    out[threadIdx.x]=1;
}
