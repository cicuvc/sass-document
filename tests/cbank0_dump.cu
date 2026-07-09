#include <cstdio>
#include <cstdint>

// 64 words = 256 bytes = fits in a single bank 0 slot on sm_90
struct Big { uint32_t a[64]; };

__global__ void fulldump(__grid_constant__ const Big b, uint32_t *out)
{
    const int base = -((int)(0x210) >> 2); // = -132
    #pragma unroll 0
    for (int i = 0; i < 132; ++i)
        out[i] = b.a[base + i];
    // also read param region to verify bank
    out[132] = b.a[0]; // should be sentinel 0 (b.a[0] sent to 0 — wait, b's initial value is what we set)
}

int main()
{
    const int count = 133; // 132 preset words + 1 sentinel
    uint32_t *d; cudaMalloc(&d, count * sizeof(uint32_t));
    Big dummy{};
    dummy.a[0] = 0xDEADBEEF; dummy.a[1] = 0xCAFEBABE;

    fulldump<<<1,1>>>(dummy, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }

    uint32_t h[133];
    cudaMemcpy(h, d, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("=== c[0x0] full preset (0x000–0x20F) H800 / sm_90  CUDA 12.8 ===\n\n");
    for (int i = 0; i < 132; i += 4) {
        int off = i * 4;
        // skip all-zero rows for readability
        bool nonzero = h[i] | h[i+1] | h[i+2] | h[i+3];
        if (!nonzero) continue;
        printf("c[0x0][0x%03x]  %08x  %08x  %08x  %08x", off,
               h[i], h[i+1], h[i+2], h[i+3]);
        if (off == 0x000) printf("  ; grid/launch config?");
        if (off == 0x010) printf("  ; ");
        if(off==0x018)printf(" ; 0=LE/RZ? 0x00007fbf → hi bits of a pointer");
        if (off == 0x020) printf("  ; pointer-like (0x7fbe…)");
        if (off == 0x028) printf("  ; [*] stack/local base (LDC R1)  [verified: 0x00fffdc0]");
        if (off == 0x030) printf("  ; ");
        if (off == 0x038) printf("  ; 0x04c0c000 = 79872000?  shared/global window?");
        if (off == 0x03c) printf("  ; 2 → CTA-per-SM?");
        if (off == 0x040) printf("  ; ");
        if (off == 0x058) printf("  ; ");
        if (off == 0x068) printf("  ; 0x120=288 → max threads/CTA?");
        if (off == 0x0c0) printf("  ; pointer-like (0x7fbe…)");
        if (off == 0x100) printf("  ; ");
        if (off == 0x10c) printf("  ; 0x72=114 → ?");
        if (off == 0x114) printf("  ; 0x400=1024 → shmem per-CTA offset");
        if (off == 0x118) printf("  ; pointer-like (0x7fbe…)");
        if (off == 0x130) printf("  ; 0x400 → another shmem offset?");
        if (off == 0x140) printf("  ; ");
        if (off == 0x150) printf("  ; 3f800000 = 1.0f × 3 → FP32 const pool?");
        if (off == 0x160) printf("  ; 0x01000000 at [0x16c] → CGA/cluster id?");
        if (off == 0x170) printf("  ; pointer-like (0x7fbf… high32)");
        if (off == 0x180) printf("  ; ");
        if (off == 0x190) printf("  ; pointer-like + offset");
        if (off == 0x1a0) printf("  ; ");
        if (off == 0x200) printf("  ; per-launch cookie [varies every run]");
        if (off == 0x208) printf("  ; [*] global mem descriptor (ULDC.64) [verified: 0x00000000_00000000]");
        printf("\n");
    }
    // sentinel check
    printf("\n-- sentinel validation --\n");
    printf("c[0x0][0x210] = 0x%08x (expect 0xdeadbeef)  %s\n",
           h[132], h[132]==0xDEADBEEF ? "OK" : "BAD BANK!");
    return 0;
}
