// TMA (cp.async.bulk.tensor) synchronization -> SASS.
// TMA load transfers a tile global->shared and signals an mbarrier via the
// transaction-count (tx) mechanism when the bytes land.
#include <cstdint>
struct TMap { char x[128]; };   // stand-in for CUtensorMap (128-byte descriptor)

extern "C" __global__ void tma_load(const __grid_constant__ TMap tmap,
                                    int cx, int cy, uint64_t* out) {
    extern __shared__ char smem[];
    __shared__ uint64_t bar;
    unsigned sbar = __cvta_generic_to_shared(&bar);
    unsigned sdst = __cvta_generic_to_shared(smem);
    const void* tm = &tmap;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(sbar));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;\n" :: "r"(sbar));
        // TMA tile load: global tensor -> shared, completes the mbarrier tx
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(sdst), "l"(tm), "r"(cx), "r"(cy), "r"(sbar) : "memory");
    }
    __syncthreads();
    // consumer: poll the mbarrier phase
    asm volatile(
        "{\n .reg .pred P1;\n"
        "L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], 0;\n"
        "@!P1 bra L;\n }\n" :: "r"(sbar));
    out[threadIdx.x] = ((uint64_t*)smem)[threadIdx.x];
}
