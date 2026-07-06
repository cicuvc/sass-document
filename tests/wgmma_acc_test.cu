// wgmma accumulator-grouping test: within one commit group, issue multiple
// wgmma.mma_async writing (a) the SAME accumulator vs (b) TWO DIFFERENT
// accumulators. Compare fence / HGMMA / group-scoreboard structure.
#include <cstdint>
#include <cuda_fp16.h>

__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
    uint64_t d = 0;
    d |= ((uint64_t)(saddr & 0x3FFFF) >> 4);
    d |= ((uint64_t)(16 >> 4)) << 16;
    d |= ((uint64_t)(16 >> 4)) << 32;
    return d;
}

#define MMA(D, DA, DB) \
  asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 " \
    "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n" \
    : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]),"+f"(D[4]),"+f"(D[5]),"+f"(D[6]),"+f"(D[7]) \
    : "l"(DA), "l"(DB))

// (a) 4 MMAs accumulating into ONE accumulator group (serial RAW on d[])
extern "C" __global__ void wgmma_same_acc(float* out, const __half* gA, const __half* gB) {
    __shared__ __half sA[64 * 16]; __shared__ __half sB[16 * 16];
    int t = threadIdx.x;
    for (int i = t; i < 64 * 16; i += blockDim.x) sA[i] = gA[i];
    for (int i = t; i < 16 * 16; i += blockDim.x) sB[i] = gB[i];
    __syncthreads();
    uint64_t dA = make_desc(__cvta_generic_to_shared(sA));
    uint64_t dB = make_desc(__cvta_generic_to_shared(sB));
    float d[8];
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n");
    MMA(d, dA, dB); MMA(d, dA, dB); MMA(d, dA, dB); MMA(d, dA, dB);   // same d
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
#pragma unroll
    for (int i = 0; i < 8; i++) out[t * 8 + i] = d[i];
}

// (b) 2 MMAs into accumulator group #0, 2 into group #1 (two independent accums)
extern "C" __global__ void wgmma_two_acc(float* out, const __half* gA, const __half* gB) {
    __shared__ __half sA[64 * 16]; __shared__ __half sB[16 * 16];
    int t = threadIdx.x;
    for (int i = t; i < 64 * 16; i += blockDim.x) sA[i] = gA[i];
    for (int i = t; i < 16 * 16; i += blockDim.x) sB[i] = gB[i];
    __syncthreads();
    uint64_t dA = make_desc(__cvta_generic_to_shared(sA));
    uint64_t dB = make_desc(__cvta_generic_to_shared(sB));
    float d0[8], d1[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { d0[i] = 0.f; d1[i] = 0.f; }
    asm volatile("wgmma.fence.sync.aligned;\n");
    MMA(d0, dA, dB); MMA(d1, dA, dB); MMA(d0, dA, dB); MMA(d1, dA, dB);  // alternate d0/d1
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
#pragma unroll
    for (int i = 0; i < 8; i++) { out[t*16+i] = d0[i]; out[t*16+8+i] = d1[i]; }
}
