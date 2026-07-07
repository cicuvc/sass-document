#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void __cluster_dims__(2,1,1) k_cluster_sync(int *a,int *out){
    cg::cluster_group cl = cg::this_cluster();
    int i=blockIdx.x*blockDim.x+threadIdx.x; a[i]+=1; cl.sync(); out[i]=a[i];
}
__global__ void __cluster_dims__(2,1,1) k_loop(int *a,int *out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    for(int k=0;k<n;k++){ a[i]+=k;
        asm volatile("barrier.cluster.arrive.release;");
        asm volatile("barrier.cluster.wait.acquire;"); }
    out[i]=a[i];
}
// plain arrive/wait for baseline
__global__ void __cluster_dims__(2,1,1) k_aw(int *a,int *out){
    int i=blockIdx.x*blockDim.x+threadIdx.x; a[i]+=1;
    asm volatile("barrier.cluster.arrive;"); asm volatile("barrier.cluster.wait;");
    out[i]=a[i];
}
