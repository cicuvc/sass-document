#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };

__global__ void probe(__grid_constant__ const Big b, int wbase, uint64_t *out)
{
    __shared__ int smem[8];
    volatile int loc[8];
    smem[threadIdx.x & 7] = threadIdx.x;
    loc[threadIdx.x & 7]  = threadIdx.x;
    __syncthreads();
    uint64_t sbase = (uint64_t)__cvta_shared_to_generic(0);          // {SWINHI,0}
    uint64_t lbase = (uint64_t)(uintptr_t)&loc[0]
                     - (unsigned)__cvta_generic_to_local((const void*)&loc[0]);
    uint32_t c18 = b.a[wbase + 0x18/4];
    uint32_t c1c = b.a[wbase + 0x1c/4];
    uint32_t c20 = b.a[wbase + 0x20/4];
    uint32_t c24 = b.a[wbase + 0x24/4];
    if (threadIdx.x == 0) {
        out[0] = sbase;
        out[1] = ((uint64_t)c1c << 32) | c18;
        out[2] = lbase;
        out[3] = ((uint64_t)c24 << 32) | c20;
    }
}

int main()
{
    uint64_t *d, h[4] = {};
    cudaMalloc(&d, sizeof h);
    Big dummy{};
    probe<<<1,8>>>(dummy, (0x00-0x210)/4, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("shared base (cvta.shared) = 0x%016llx\n", (unsigned long long)h[0]);
    printf("{c[0x1c],c[0x18]}          = 0x%016llx  %s\n", (unsigned long long)h[1],
           h[0]==h[1] ? "== MATCH" : "!= differ");
    printf("local  base (cvta.local)  = 0x%016llx\n", (unsigned long long)h[2]);
    printf("{c[0x24],c[0x20]}          = 0x%016llx  %s\n", (unsigned long long)h[3],
           h[2]==h[3] ? "== MATCH" : "!= differ");
    return 0;
}
