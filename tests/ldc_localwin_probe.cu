#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };

__global__ void probe(__grid_constant__ const Big b, int wbase, uint64_t *out)
{
    // local array -> generic address forces cvta.local (materializes local window base)
    volatile int loc[16];
    loc[threadIdx.x & 15] = (int)threadIdx.x;
    uint64_t gen_local = (uint64_t)(uintptr_t)((const void*)&loc[0]);
    unsigned loc_off   = (unsigned)__cvta_generic_to_local((const void*)&loc[0]);

    // read preset words via register-indexed LDC (wbase = (0x00-0x210)/4 at runtime)
    uint32_t c20 = b.a[wbase + 0x20/4];
    uint32_t c24 = b.a[wbase + 0x24/4];
    uint32_t c28 = b.a[wbase + 0x28/4];
    uint32_t c44 = b.a[wbase + 0x44/4];
    uint32_t c48 = b.a[wbase + 0x48/4];
    uint32_t c6c = b.a[wbase + 0x6c/4];

    if (threadIdx.x == 0) {
        out[0] = gen_local;
        out[1] = loc_off;
        out[2] = ((uint64_t)c24 << 32) | c20;   // candidate local window base
        out[3] = c28;
        out[4] = ((uint64_t)c48 << 32) | c44;
        out[5] = c6c;
    }
}

int main()
{
    uint64_t *d, h[6] = {};
    cudaMalloc(&d, sizeof h);
    Big dummy{};
    int wbase = (0x00 - 0x210) / 4;
    probe<<<1, 4>>>(dummy, wbase, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("generic &loc[0]        = 0x%016llx\n", (unsigned long long)h[0]);
    printf("local-space offset     = 0x%08llx\n",  (unsigned long long)h[1]);
    printf("implied local win base = 0x%016llx\n", (unsigned long long)(h[0]-h[1]));
    printf("{c[0x24],c[0x20]}       = 0x%016llx\n", (unsigned long long)h[2]);
    printf("c[0x0][0x28]           = 0x%08llx\n",  (unsigned long long)h[3]);
    printf("{c[0x48],c[0x44]}       = 0x%016llx\n", (unsigned long long)h[4]);
    printf("c[0x0][0x6c]           = 0x%08llx\n",  (unsigned long long)h[5]);
    return 0;
}
