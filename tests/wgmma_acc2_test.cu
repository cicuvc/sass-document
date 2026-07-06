// wgmma follow-ups:
// (1) same accumulator, DIFFERENT inputs per MMA (advancing descriptors = K-loop)
// (2) large-n wgmma (m64n64/m64n128) -> accumulator spans many register groups
#include <cstdint>
#include <cuda_fp16.h>

__device__ __forceinline__ uint64_t md(uint32_t s) {
    uint64_t d = 0;
    d |= ((uint64_t)(s & 0x3FFFF) >> 4);
    d |= ((uint64_t)1) << 16;
    d |= ((uint64_t)1) << 32;
    return d;
}

#define MMA16(D,DA,DB) asm volatile( \
  "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n" \
  : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
  : "l"(DA), "l"(DB))

// (1) same accumulator d[8], four MMAs over four DIFFERENT shared-memory K-tiles
extern "C" __global__ void same_acc_diff_input(float* out, const __half* gA, const __half* gB) {
    __shared__ __half sA[64 * 16 * 4];
    __shared__ __half sB[16 * 16 * 4];
    int t = threadIdx.x;
    for (int i = t; i < 64 * 16 * 4; i += blockDim.x) sA[i] = gA[i];
    for (int i = t; i < 16 * 16 * 4; i += blockDim.x) sB[i] = gB[i];
    __syncthreads();
    unsigned baseA = __cvta_generic_to_shared(sA);
    unsigned baseB = __cvta_generic_to_shared(sB);
    float d[8];
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n");
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA = md(baseA + k * (64 * 16 * 2));   // advance A tile (2 bytes/half)
        uint64_t dB = md(baseB + k * (16 * 16 * 2));   // advance B tile
        MMA16(d, dA, dB);                              // SAME d, different inputs
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
#pragma unroll
    for (int i = 0; i < 8; i++) out[t * 8 + i] = d[i];
}

// (2) large n: m64n64k16 -> 32 accumulator regs/thread (4 register groups of 8)
extern "C" __global__ void large_n64(float* out, const __half* gA, const __half* gB) {
    __shared__ __half sA[64 * 16];
    __shared__ __half sB[64 * 16];
    int t = threadIdx.x;
    for (int i = t; i < 64 * 16; i += blockDim.x) { sA[i] = gA[i]; sB[i] = gB[i]; }
    __syncthreads();
    uint64_t dA = md(__cvta_generic_to_shared(sA));
    uint64_t dB = md(__cvta_generic_to_shared(sB));
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n");
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "%32, %33, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
        "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
        "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
        "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
      : "l"(dA), "l"(dB));
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
#pragma unroll
    for (int i = 0; i < 32; i++) out[t * 32 + i] = d[i];
}
