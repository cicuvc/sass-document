// cp.async.bulk commit_group / wait_group (bulk-async-group completion) -> SASS.
// The group mechanism (vs mbarrier tx-count) is used mainly for TMA STORES
// (shared->global) where there is no consumer mbarrier to signal.
#include <cstdint>
struct TMap { char x[128]; };

extern "C" __global__ void tma_store(const __grid_constant__ TMap tmap, int cx, int cy) {
    extern __shared__ char smem[];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) smem[i] = (char)i;
    __syncthreads();
    unsigned ssrc = __cvta_generic_to_shared(smem);
    const void* tm = &tmap;
    if (threadIdx.x == 0) {
        // TMA tile store shared -> global, into a bulk async-group
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
            :: "l"(tm), "r"(cx), "r"(cy), "r"(ssrc) : "memory");
        asm volatile("cp.async.bulk.commit_group;\n");
        asm volatile("cp.async.bulk.wait_group.read 0;\n");   // wait all bulk groups
    }
}
