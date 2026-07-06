// wgmma fp8 multi-warpgroup hazard probe v2 (Hopper sm_90a) -- artifact-hardened.
// Each warpgroup owns a large (8 KB) shared slice FULLY filled with ITS dataset's
// pattern, so even if the (deliberately crude) descriptor over-reads, it can only
// read its own dataset -- never the neighbor's. The reference (1 warpgroup) fills
// its slice identically. Tensor core is deterministic => concurrent result must be
// bit-identical to the isolated reference; any diff at wgs>K = real cross-warpgroup
// accumulator hazard.
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define DATASETS 2048
#define SLICE 8192          // bytes per warpgroup (>> wgmma read range)

__device__ __forceinline__ uint64_t make_desc(uint32_t s) {
    uint64_t d = 0;
    d |= ((uint64_t)(s & 0x3FFFF) >> 4);
    d |= ((uint64_t)1) << 16;
    d |= ((uint64_t)1) << 32;
    return d;
}
__device__ __forceinline__ __nv_fp8_e4m3 gen(unsigned ds, int i) {
    unsigned x = ds * 2654435761u + (unsigned)i * 40503u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return __nv_fp8_e4m3(((int)(x & 0xff) - 128) * (1.0f / 64.0f));
}

#define MMA(d,dA,dB) asm volatile( \
  "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1;\n" \
  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) \
  : "l"(dA), "l"(dB))

__device__ __forceinline__ void compute(unsigned ds, int M, float d[8]) {
    extern __shared__ char sm[];
    int wg = threadIdx.x >> 7;
    __nv_fp8_e4m3* base = (__nv_fp8_e4m3*)(sm + wg * SLICE);
    int lane = threadIdx.x & 127;
    for (int i = lane; i < SLICE; i += 128) base[i] = gen(ds, i);   // fill WHOLE slice
    __syncthreads();
    uint64_t dA = make_desc(__cvta_generic_to_shared(base));
    uint64_t dB = make_desc(__cvta_generic_to_shared(base + 2048));
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n");
    for (int k = 0; k < M; k++) MMA(d, dA, dB);
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

extern "C" __global__ void ref_kernel(int M, float* ref) {
    unsigned ds = blockIdx.x;
    float d[8]; compute(ds, M, d);
    int lane = threadIdx.x & 127;
    float* r = ref + (size_t)ds * 128 * 8 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) r[i] = d[i];
}
extern "C" __global__ void test_kernel(int M, int N, const float* ref,
                                       unsigned long long* mm, float* maxdev) {
    int wg = threadIdx.x >> 7;
    unsigned ds = ((unsigned)blockIdx.x * N + wg) % DATASETS;
    float d[8]; compute(ds, M, d);
    int lane = threadIdx.x & 127;
    const float* r = ref + (size_t)ds * 128 * 8 + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; i++)
        if (__float_as_uint(d[i]) != __float_as_uint(r[i])) {
            atomicAdd(mm, 1ULL);
            atomicMax((int*)maxdev, __float_as_int(fabsf(d[i] - r[i])));
        }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 64;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    int smem = 8 * SLICE;
    cudaFuncSetAttribute(test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(ref_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, SLICE);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    float* ref; cudaMalloc(&ref, (size_t)DATASETS * 128 * 8 * 4);
    unsigned long long* d_mm; float* d_max;
    cudaMalloc(&d_mm, 8); cudaMalloc(&d_max, 4);

    ref_kernel<<<DATASETS, 128, SLICE>>>(M, ref);
    printf("H800 fp8 wgmma hazard v2  M=%d reps=%d SMs=%d  ref=%s\n",
           M, reps, nsm, cudaGetErrorString(cudaDeviceSynchronize()));
    for (int N = 1; N <= 8; N++) {
        int block = N * 128; if (block > 1024) break;
        int grid = nsm * 2;
        unsigned long long tot = 0; float gmax = 0;
        for (int r = 0; r < reps; r++) {
            cudaMemset(d_mm, 0, 8); cudaMemset(d_max, 0, 4);
            test_kernel<<<grid, block, N * SLICE>>>(M, N, ref, d_mm, d_max);
            if (cudaDeviceSynchronize() != cudaSuccess) { printf("  N=%d ERR %s\n", N, cudaGetErrorString(cudaGetLastError())); break; }
            unsigned long long mm; float mx;
            cudaMemcpy(&mm, d_mm, 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(&mx, d_max, 4, cudaMemcpyDeviceToHost);
            tot += mm; if (mx > gmax) gmax = mx;
        }
        double checked = (double)reps * grid * block * 8.0;
        printf("  wgs=%d (%d wg/SM)  mismatches=%llu / %.0f (%.2e)  maxdev=%g%s\n",
               N, N, tot, checked, tot / checked, gmax, tot ? "   <-- HAZARD" : "");
    }
    return 0;
}
