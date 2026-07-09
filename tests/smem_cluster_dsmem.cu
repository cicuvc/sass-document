#include <cstdio>
#include <cstdint>

__global__ void __cluster_dims__(2, 1, 1) ck(uint64_t *out)
{
    __shared__ int smem[4];
    smem[0] = blockIdx.x;
    __syncthreads();
    unsigned off = (unsigned)__cvta_generic_to_shared(&smem[0]); // (rank<<24)+0x400 ?
    if (threadIdx.x == 0) out[blockIdx.x] = off;
}

int main()
{
    uint64_t *d, h[2] = {};
    cudaMalloc(&d, sizeof h);
    ck<<<2, 1>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("CTA0 shared offset = 0x%08llx\n", (unsigned long long)h[0]);
    printf("CTA1 shared offset = 0x%08llx\n", (unsigned long long)h[1]);
    printf("delta              = 0x%08llx (expect 0x1000000 = 1<<24)\n",
           (unsigned long long)(h[1] - h[0]));
    return 0;
}
