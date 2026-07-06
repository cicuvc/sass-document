// wgmma (warpgroup async MMA) synchronization test.
// Standard pipeline: wgmma.fence -> N x wgmma.mma_async -> commit_group -> wait_group.
// Needs sm_90a. Accumulator m64n16k16 f32 = 8 f32 regs/thread.
#include <cstdint>
#include <cuda_fp16.h>

__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
    // minimal shared-memory matrix descriptor (address + default 16B strides)
    uint64_t d = 0;
    d |= ((uint64_t)(saddr & 0x3FFFF) >> 4);      // start addr [13:0]
    d |= ((uint64_t)(16 >> 4)) << 16;             // leading-dim byte offset
    d |= ((uint64_t)(16 >> 4)) << 32;             // stride-dim byte offset
    return d;
}

extern "C" __global__ void wgmma_pipe(float* out, const __half* gA, const __half* gB) {
    __shared__ __half sA[64 * 16];
    __shared__ __half sB[16 * 16];
    int t = threadIdx.x;
    // load operands into shared (not the focus; just to have real descriptors)
    for (int i = t; i < 64 * 16; i += blockDim.x) sA[i] = gA[i];
    for (int i = t; i < 16 * 16; i += blockDim.x) sB[i] = gB[i];
    __syncthreads();

    uint64_t descA = make_desc(__cvta_generic_to_shared(sA));
    uint64_t descB = make_desc(__cvta_generic_to_shared(sB));

    float d[8];
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 0.f;

    // ---- the pipeline under study ----
    asm volatile("wgmma.fence.sync.aligned;\n");                 // protect accumulator regs
    // two async MMAs accumulating into d[]
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
      : "l"(descA), "l"(descB));
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
      : "l"(descA), "l"(descB));
    asm volatile("wgmma.commit_group.sync.aligned;\n");          // commit the batch
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");          // wait until done
    // ----------------------------------

#pragma unroll
    for (int i = 0; i < 8; i++) out[t * 8 + i] = d[i];
}
