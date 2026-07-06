// wgmma multi-warpgroup hazard probe (Hopper sm_90a).
// All-ones A/B -> each m64n16k16 wgmma yields exactly 16.0 per element,
// independent of shared-memory layout (permuting ones is still ones).
// After M accumulations every accumulator element must equal 16*M exactly.
// We run WGS warpgroups per block (all hammering the SM tensor core together)
// and count any deviation -> tests the ">3 warpgroups -> random precision loss" rumor.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ uint64_t make_desc(uint32_t s) {
    uint64_t d = 0;
    d |= ((uint64_t)(s & 0x3FFFF) >> 4);
    d |= ((uint64_t)1) << 16;   // leading-dim byte offset (/16)
    d |= ((uint64_t)1) << 32;   // stride-dim byte offset (/16)
    return d;
}

#define MMA(d,dA,dB) asm volatile( \
  "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n" \
  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) \
  : "l"(dA), "l"(dB))

extern "C" __global__ void wg_hazard(int M, unsigned long long* mismatches,
                                     float* first_bad, float* sample) {
    extern __shared__ __half smem[];               // filled with 1.0
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem[i] = __float2half(1.0f);
    __syncthreads();
    // each warpgroup uses its own slice of shared as A/B (all ones anyway)
    uint64_t dA = make_desc(__cvta_generic_to_shared(smem));
    uint64_t dB = make_desc(__cvta_generic_to_shared(smem + 1024));
    float d[8];
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n");
    for (int k = 0; k < M; k++) { MMA(d, dA, dB); }
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");

    float expect = 16.0f * (float)M;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (d[i] != expect) {
            unsigned long long n = atomicAdd(mismatches, 1ULL);
            if (n == 0) *first_bad = d[i];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *sample = d[0];
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    int smemBytes = 4096 * sizeof(__half);
    cudaFuncSetAttribute(wg_hazard, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);

    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    unsigned long long *d_mm; float *d_fb, *d_s;
    cudaMalloc(&d_mm, 8); cudaMalloc(&d_fb, 4); cudaMalloc(&d_s, 4);

    printf("H800 wgmma hazard probe  M=%d reps=%d SMs=%d  expect=%.1f\n", M, reps, nsm, 16.0*M);
    for (int wgs = 1; wgs <= 8; wgs++) {
        int block = wgs * 128;
        if (block > 1024) break;
        int grid = nsm * 2;                        // oversubscribe SMs
        unsigned long long total_mm = 0; float fb = 0, samp = 0;
        for (int r = 0; r < reps; r++) {
            cudaMemset(d_mm, 0, 8);
            wg_hazard<<<grid, block, smemBytes>>>(M, d_mm, d_fb, d_s);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf("  wgs=%d LAUNCH ERROR: %s\n", wgs, cudaGetErrorString(e)); break; }
            unsigned long long mm; cudaMemcpy(&mm, d_mm, 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(&fb, d_fb, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&samp, d_s, 4, cudaMemcpyDeviceToHost);
            total_mm += mm;
        }
        double checked = (double)reps * grid * block * 8.0;
        printf("  wgs=%d (block=%4d, %d wg/SM x %d)  sample=%.1f  mismatches=%llu / %.0f  (%.2e)%s\n",
               wgs, block, wgs, grid, samp, total_mm, checked,
               checked>0? total_mm/checked : 0.0,
               total_mm? "   <-- HAZARD" : "");
    }
    cudaFree(d_mm); cudaFree(d_fb); cudaFree(d_s);
    return 0;
}
