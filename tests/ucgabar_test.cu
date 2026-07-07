#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void __cluster_dims__(2,1,1) k(int *a,int *out){
    cg::cluster_group cl = cg::this_cluster();
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i]+=1;
    cl.barrier_arrive();          // arrive on cluster barrier
    cl.barrier_wait(cl.barrier_arrive());
    out[i]=a[i];
}
