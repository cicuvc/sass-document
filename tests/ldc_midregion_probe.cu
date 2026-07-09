#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };
extern __shared__ int dyn[];

__global__ void __cluster_dims__(2,1,1)
probe(__grid_constant__ const Big b, int wbase, int count, uint32_t *out)
{
    dyn[threadIdx.x] = threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        #pragma unroll 1
        for (int i = 0; i < count; ++i)
            out[i] = b.a[wbase + i];
    }
}

int main()
{
    const int start = 0x00, end = 0x80, count = (end-start)/4;
    uint32_t *d; cudaMalloc(&d, count*sizeof(uint32_t));
    Big dummy{};
    int wbase = (start - 0x210)/4;
    cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);
    probe<<<4, 32, 3072>>>(dummy, wbase, count, d);   // dyn smem = 0xC00 bytes
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    uint32_t *h = new uint32_t[count];
    cudaMemcpy(h, d, count*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i=0;i<count;i++) if (h[i]) printf("c[0x0][0x%02x] = 0x%08x\n", start+i*4, h[i]);
    return 0;
}
