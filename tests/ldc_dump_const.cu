#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[64]; };

// b.a[] is placed at c[0x0][0x210]. b.a[idx] == c[0x0][0x210 + idx*4].
// A negative idx reaches the driver-preset region below the param base.
__global__ void dumpc(__grid_constant__ const Big b, int base_word, int count, uint32_t *out)
{
    #pragma unroll 1
    for (int i = 0; i < count; ++i)
        out[i] = b.a[base_word + i];
}

int main()
{
    const int PARAM_BASE = 0x210;
    const int start_off  = 0x00;     // first byte offset to dump
    const int end_off    = 0x230;    // exclusive
    const int count      = (end_off - start_off) / 4;
    const int base_word  = (start_off - PARAM_BASE) / 4;

    uint32_t *d; cudaMalloc(&d, count * sizeof(uint32_t));
    Big dummy{}; // sentinels land at 0x210.. to validate the offset math
    dummy.a[0] = 0xDEADBEEF; dummy.a[1] = 0xCAFEBABE;
    dummy.a[2] = 0x12345678; dummy.a[3] = 0xA5A5A5A5;
    dumpc<<<1,1>>>(dummy, base_word, count, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("launch err: %s\n", cudaGetErrorString(e)); return 1; }

    uint32_t *h = new uint32_t[count];
    cudaMemcpy(h, d, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("=== c[0x0] preset region dump (H800 / sm_90) ===\n");
    for (int i = 0; i < count; ++i) {
        int off = start_off + i * 4;
        const char *tag = "";
        if (off == 0x28)  tag = "  <- stack/local base -> R1";
        if (off == 0x208) tag = "  <- global mem desc (lo)";
        if (off == 0x20c) tag = "  <- global mem desc (hi)";
        if (off == 0x210) tag = "  <- param base";
        printf("c[0x0][0x%03x] = 0x%08x%s\n", off, h[i], tag);
    }
    // also print the descriptor as a 64-bit value
    int di = (0x208 - start_off) / 4;
    printf("\nglobal mem descriptor @0x208 = 0x%08x%08x\n", h[di+1], h[di]);
    int si = (0x28 - start_off) / 4;
    printf("stack base            @0x28  = 0x%08x%08x\n", h[si+1], h[si]);
    return 0;
}
