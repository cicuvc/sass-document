// TMA / mbarrier synchronization study.
// (1) the classic mbarrier.try_wait.parity polling loop
// (2) mbarrier init / arrive / expect_tx  (producer side)
// (3) cp.async.bulk.tensor (TMA load) driving an mbarrier
#include <cstdint>
#include <cuda/barrier>

extern "C" __global__ void mbar_try_wait(uint64_t* g, int phase) {
    __shared__ uint64_t bar;
    unsigned mbar_ptr = __cvta_generic_to_shared(&bar);
    unsigned kPhaseBit = phase;
    // classic poll loop
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(kPhaseBit)
    );
    g[threadIdx.x] = bar;
}

// producer: init + arrive + expect_tx
extern "C" __global__ void mbar_ops(uint64_t* g) {
    __shared__ uint64_t bar;
    unsigned p = __cvta_generic_to_shared(&bar);
    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(p));
    __syncthreads();
    uint64_t tok;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], 128;\n"
                 : "=l"(tok) : "r"(p));
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n" : "=l"(tok) : "r"(p));
    g[threadIdx.x] = tok;
}
