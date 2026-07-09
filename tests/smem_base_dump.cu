#include <cstdio>
#include <cstdint>

__global__ void probe(uint64_t *out)
{
    __shared__ int smem[8];
    smem[threadIdx.x & 7] = threadIdx.x;
    __syncthreads();
    // shared-window offset (what STS/LDS use) vs generic address
    unsigned smem_off = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // shared->generic on offset 0 materializes the pure shared window base
    uint64_t base0    = (uint64_t)__cvta_shared_to_generic(0);
    uint64_t gen      = (uint64_t)__cvta_shared_to_generic(smem_off);
    if (threadIdx.x == 0) {
        out[0] = gen;                 // generic address of smem[0]
        out[1] = smem_off;            // shared-space offset of smem[0]
        out[2] = base0;               // generic address of shared offset 0 (window base)
    }
}

int main()
{
    uint64_t *d, h[3] = {};
    cudaMalloc(&d, sizeof h);
    probe<<<1, 8>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("generic &smem[0]      = 0x%016llx\n", (unsigned long long)h[0]);
    printf("shared-window offset  = 0x%08llx\n",  (unsigned long long)h[1]);
    printf("shared base (off 0)   = 0x%016llx\n", (unsigned long long)h[2]);
    printf("implied SWIN base     = 0x%016llx\n", (unsigned long long)(h[0] - h[1]));
    return 0;
}
