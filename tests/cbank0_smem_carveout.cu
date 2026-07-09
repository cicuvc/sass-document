#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };
#define NW 132
extern __shared__ int esh[];

__global__ void probe(__grid_constant__ const Big b, int wbase, uint32_t *out)
{
    if (threadIdx.x) return;
    #pragma unroll 1
    for (int i = 0; i < NW; ++i) out[i] = b.a[wbase + i];
}

int main()
{
    uint32_t *d; cudaMalloc(&d, NW*sizeof(uint32_t));
    Big dummy{};
    int wbase = (0x00-0x210)/4;

    // allow large dynamic shared memory
    cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024);

    struct Cfg { const char* tag; int dynKB; int carveout; };
    Cfg cfgs[] = {
        {"base dyn=0 cv=def",   0,   cudaSharedmemCarveoutDefault},
        {"dyn=32K  cv=def",     32,  cudaSharedmemCarveoutDefault},
        {"dyn=100K cv=def",     100, cudaSharedmemCarveoutDefault},
        {"dyn=200K cv=def",     200, cudaSharedmemCarveoutDefault},
        {"dyn=0    cv=MaxL1",   0,   cudaSharedmemCarveoutMaxL1},
        {"dyn=0    cv=MaxShared",0,  cudaSharedmemCarveoutMaxShared},
        {"dyn=0    cv=50",      0,   50},
        {"dyn=8K   cv=MaxL1",   8,   cudaSharedmemCarveoutMaxL1},
        {"dyn=8K   cv=MaxShared",8,  cudaSharedmemCarveoutMaxShared},
    };
    const int NC = sizeof(cfgs)/sizeof(cfgs[0]);
    uint32_t cols[NC][NW];

    for (int c=0;c<NC;c++){
        cudaFuncSetAttribute(probe, cudaFuncAttributePreferredSharedMemoryCarveout, cfgs[c].carveout);
        cudaMemset(d,0,NW*sizeof(uint32_t));
        probe<<<1, 64, cfgs[c].dynKB*1024>>>(dummy, wbase, d);
        cudaError_t e = cudaDeviceSynchronize();
        if (e){ printf("%-22s ERR %s\n", cfgs[c].tag, cudaGetErrorString(e)); 
                for(int i=0;i<NW;i++) cols[c][i]=0xffffffff; continue; }
        cudaMemcpy(cols[c], d, NW*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    printf("%-6s", "off");
    for (int c=0;c<NC;c++) printf(" %18s", cfgs[c].tag);
    printf("\n");
    for (int i=0;i<NW;i++){
        bool show=false;
        for (int c=0;c<NC;c++) if (cols[c][i]!=cols[0][i]) show=true;
        if (!show) continue;
        printf("0x%03x", i*4);
        for (int c=0;c<NC;c++) printf(" %18x", cols[c][i]);
        printf("\n");
    }
    // always show the known shared-related slots
    printf("\n-- known shared slots (all configs) --\n");
    for (int off : {0x2c,0x114,0x13c,0x16c}) {
        printf("0x%03x", off);
        for (int c=0;c<NC;c++) printf(" %18x", cols[c][off/4]);
        printf("\n");
    }
    return 0;
}
