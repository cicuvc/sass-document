#include <cooperative_groups.h>
#include <utility>
namespace cg = cooperative_groups;
// CG token: arrive returns a token, wait consumes it (token round-trip may need GET/SET)
__global__ void __cluster_dims__(2,1,1) k_token(int *a, int *out) {
    cg::cluster_group cl = cg::this_cluster();
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    auto tok = cl.barrier_arrive();
    a[i] += 1;
    cl.barrier_wait(std::move(tok));
    out[i] = a[i];
}
// mapa / cluster mapped shared mem + mbarrier init/arrive (may emit SET/GET)
#include <cuda/barrier>
__global__ void __cluster_dims__(2,1,1) k_mbar(int *out) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();
    auto tok = bar.arrive();
    bar.wait(std::move(tok));
    out[threadIdx.x] = 1;
}
