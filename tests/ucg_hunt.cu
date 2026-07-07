#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2,1,1) k_mbar_cluster(int *out) {
    __shared__ unsigned long long bar;
    unsigned smem = (unsigned)__cvta_generic_to_shared(&bar);
    if (threadIdx.x == 0) asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem), "r"(blockDim.x));
    __syncthreads();
    unsigned long long tok;
    asm volatile("mbarrier.arrive.release.cluster.b64 %0, [%1];" : "=l"(tok) : "r"(smem));
    out[threadIdx.x] = (int)tok;
}
__global__ void __cluster_dims__(2,1,1) k_cluster_sync(int *a,int *out){
    cg::cluster_group cl = cg::this_cluster();
    int i=blockIdx.x*blockDim.x+threadIdx.x; a[i]+=1; cl.sync(); out[i]=a[i];
}
__global__ void __cluster_dims__(2,1,1) k_dsmem(int *out){
    cg::cluster_group cl = cg::this_cluster();
    __shared__ unsigned long long bar;
    if (threadIdx.x==0){ unsigned s=(unsigned)__cvta_generic_to_shared(&bar);
        asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(s),"r"(blockDim.x)); }
    cl.sync();
    unsigned long long *remote = cl.map_shared_rank(&bar, cl.block_rank()^1);
    unsigned rs=(unsigned)__cvta_generic_to_shared(remote);
    unsigned long long tok;
    asm volatile("mbarrier.arrive.release.cluster.b64 %0,[%1];":"=l"(tok):"r"(rs));
    out[threadIdx.x]=(int)tok;
}
__global__ void __cluster_dims__(2,1,1) k_loop(int *a,int *out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    for(int k=0;k<n;k++){ a[i]+=k;
        asm volatile("barrier.cluster.arrive.release;");
        asm volatile("barrier.cluster.wait.acquire;"); }
    out[i]=a[i];
}
