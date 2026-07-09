#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };
#define NW 132

__global__ void probe(__grid_constant__ const Big b, int wbase, uint32_t *out)
{
    if (threadIdx.x) return;
    #pragma unroll 1
    for (int i = 0; i < NW; ++i) out[i] = b.a[wbase + i];
}

int main()
{
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    size_t freeB=0, totB=0; cudaMemGetInfo(&freeB, &totB);

    int clkSM=0, clkMem=0, l2=0, smem_sm=0, regs_sm=0, thr_sm=0, persistL2=0, busW=0;
    cudaDeviceGetAttribute(&clkSM,  cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&clkMem, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&l2,     cudaDevAttrL2CacheSize, dev);
    cudaDeviceGetAttribute(&smem_sm,cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&regs_sm,cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    cudaDeviceGetAttribute(&persistL2, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&busW,   cudaDevAttrGlobalMemoryBusWidth, dev);

    printf("=== device props (%s) ===\n", p.name);
    printf("SM count           = %d (0x%x)\n", p.multiProcessorCount, p.multiProcessorCount);
    printf("SM clock  (kHz)    = %d (0x%x)\n", clkSM, clkSM);
    printf("mem clock (kHz)    = %d (0x%x)\n", clkMem, clkMem);
    printf("SM clock  (Hz)     = %lld (0x%llx)\n",(long long)clkSM*1000,(long long)clkSM*1000);
    printf("totalGlobalMem     = %zu (0x%zx)\n", (size_t)p.totalGlobalMem, (size_t)p.totalGlobalMem);
    printf("mem free / total   = %zu / %zu\n", freeB, totB);
    printf("  free  KiB=%zu MiB=%zu\n", freeB>>10, freeB>>20);
    printf("  total KiB=%zu MiB=%zu\n", totB>>10, totB>>20);
    printf("L2 cache size      = %d (0x%x)\n", l2, l2);
    printf("persist L2 max     = %d (0x%x)\n", persistL2, persistL2);
    printf("smem/SM            = %d (0x%x)\n", smem_sm, smem_sm);
    printf("regs/SM            = %d (0x%x)\n", regs_sm, regs_sm);
    printf("threads/SM         = %d (0x%x)\n", thr_sm, thr_sm);
    printf("mem bus width      = %d\n", busW);

    uint32_t *d; cudaMalloc(&d, NW*sizeof(uint32_t));
    Big dummy{};
    probe<<<1,1>>>(dummy, (0x00-0x210)/4, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    uint32_t *h = new uint32_t[NW];
    cudaMemcpy(h, d, NW*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("\n=== preset region nonzero slots (offset : hex : dec) ===\n");
    for (int i=0;i<NW;i++) if (h[i]) printf("0x%03x : 0x%08x : %u\n", i*4, h[i], h[i]);
    return 0;
}
