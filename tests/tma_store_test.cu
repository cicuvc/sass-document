#include <cstdint>
#include <cuda_runtime.h>
// Producer side: shared -> global bulk copy (TMA store), bulk async-group
// completion. Focus on the fence pattern BEFORE the store-out and around
// commit_group / wait_group.
extern "C" __global__ void tma_store(void* gdst, const uint32_t* in, uint32_t nbytes){
    extern __shared__ uint32_t smem[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    // fill shared via generic-proxy stores
    smem[threadIdx.x] = in[threadIdx.x] + 1;
    __syncthreads();
    if(threadIdx.x==0){
        // generic-proxy writes must be visible to async proxy before the bulk store
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
            ::"l"(gdst),"r"(smem_addr),"r"(nbytes):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    }
}
// Also: plain cp.async (Ampere, generic proxy) commit/wait for contrast
extern "C" __global__ void cpasync_generic(const void* g, uint32_t* out){
    extern __shared__ uint32_t smem[];
    uint32_t smem_addr=(uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(smem_addr),"l"(g):"memory");
    asm volatile("cp.async.commit_group;":::"memory");
    asm volatile("cp.async.wait_group 0;":::"memory");
    __syncthreads();
    out[threadIdx.x]=smem[threadIdx.x];
}
