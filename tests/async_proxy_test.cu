#include <cstdint>
#include <cuda_runtime.h>
// SASS for async-proxy fences and cp.async.bulk (TMA).

// --- fence.proxy.async variants (standalone) ---
extern "C" __global__ void fpa_plain(uint32_t*o){ asm volatile("fence.proxy.async;":::"memory"); *o=1; }
extern "C" __global__ void fpa_global(uint32_t*o){ asm volatile("fence.proxy.async.global;":::"memory"); *o=1; }
extern "C" __global__ void fpa_shared_cta(uint32_t*o){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); *o=1; }
extern "C" __global__ void fpa_shared_cluster(uint32_t*o){ asm volatile("fence.proxy.async.shared::cluster;":::"memory"); *o=1; }
// contrast: generic fence.proxy.alias
extern "C" __global__ void fp_alias(uint32_t*o){ asm volatile("fence.proxy.alias;":::"memory"); *o=1; }

// --- cp.async.bulk TMA global->shared::cta with mbarrier, then fence + read ---
extern "C" __global__ void tma_load(const void* gsrc, uint32_t* out, uint32_t nbytes){
    extern __shared__ uint32_t smem[];
    __shared__ uint64_t mbar;
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
    if(threadIdx.x==0){
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(mbar_addr):"memory");
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        // bulk copy global -> shared, mbarrier completion
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
            ::"r"(smem_addr),"l"(gsrc),"r"(nbytes),"r"(mbar_addr):"memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(mbar_addr),"r"(nbytes):"memory");
    }
    __syncthreads();
    // wait loop
    uint32_t done=0;
    while(!done){
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
            :"=r"(done):"r"(mbar_addr):"memory");
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");   // async-proxy -> generic-proxy before reading
    out[threadIdx.x] = smem[threadIdx.x];
}
