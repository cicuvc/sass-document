// mbarrier.arrive / expect_tx variants -> SASS (SYNCS.ARRIVE.* family).
#include <cstdint>
extern "C" __global__ void mbar_arrive_variants(uint64_t* g) {
    __shared__ uint64_t bar;
    unsigned p = __cvta_generic_to_shared(&bar);
    uint64_t t0, t1; uint32_t cnt = g[0];
    // 1. plain arrive (count 1), returns token
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n" : "=l"(t0) : "r"(p));
    // 2. arrive with explicit count
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;\n" : "=l"(t1) : "r"(p), "r"(cnt));
    // 3. standalone expect_tx
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" :: "r"(p), "r"(cnt));
    // 4. combined arrive.expect_tx
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 256;\n" :: "r"(p));
    // 5. arrive with cluster scope + relaxed
    asm volatile("mbarrier.arrive.release.cluster.shared::cta.b64 _, [%0];\n" :: "r"(p));
    // 6. arrive to a remote cta in cluster (mapa)
    g[threadIdx.x] = t0 + t1;
}
