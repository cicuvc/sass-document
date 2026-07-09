#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };

// b.a[] sits at c[0x0][0x210]; negative index reaches the preset region.
__global__ void probe(__grid_constant__ const Big b, uint32_t *out)
{
    if (threadIdx.x || blockIdx.x || threadIdx.y || blockIdx.y) return;
    #pragma unroll 1
    for (int i = 0; i < 6; ++i)
        out[i] = b.a[(0x00 - 0x210) / 4 + i];  // reads c[0x0][0x00 + i*4]
    out[6] = blockDim.x; out[7] = blockDim.y; out[8] = blockDim.z;
    out[9] = gridDim.x;  out[10]= gridDim.y;  out[11]= gridDim.z;
}

int main()
{
    uint32_t *d, h[12] = {};
    cudaMalloc(&d, sizeof h);
    Big dummy{};
    dim3 block(2,4,6), grid(3,5,7);
    probe<<<grid, block>>>(dummy, d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    const char* n[6] = {"0x00","0x04","0x08","0x0c","0x10","0x14"};
    for (int i=0;i<6;i++) printf("c[0x0][%s] = %u\n", n[i], h[i]);
    printf("blockDim = (%u,%u,%u)  gridDim = (%u,%u,%u)\n",
           h[6],h[7],h[8],h[9],h[10],h[11]);
    return 0;
}
