#include <cstdio>
#include <cstdint>
#include <cooperative_groups.h>

struct Big { uint32_t a[4]; };
#define NW 132                     // words 0x00 .. 0x20c

__global__ void probe(__grid_constant__ const Big b, int wbase, uint32_t *out)
{
    if (blockIdx.x|blockIdx.y|blockIdx.z|threadIdx.x|threadIdx.y|threadIdx.z) return;
    #pragma unroll 1
    for (int i = 0; i < NW; ++i) out[i] = b.a[wbase + i];
}

int main()
{
    uint32_t *d; cudaMalloc(&d, NW * sizeof(uint32_t));
    Big dummy{};
    int wbase = (0x00 - 0x210) / 4;

    struct Cfg { const char* tag; dim3 g, b; size_t dyn; int cx, cy, cz; bool coop; };
    Cfg cfgs[] = {
        {"base<<<1,1>>>",        {1,1,1},   {1,1,1},   0,   0,0,0, false},
        {"block(256,1,1)",       {1,1,1},   {256,1,1}, 0,   0,0,0, false},
        {"block(1,7,1)",         {1,1,1},   {1,7,1},   0,   0,0,0, false},
        {"block(1,1,5)",         {1,1,1},   {1,1,5},   0,   0,0,0, false},
        {"grid(11,1,1)",         {11,1,1},  {1,1,1},   0,   0,0,0, false},
        {"grid(1,13,1)",         {1,13,1},  {1,1,1},   0,   0,0,0, false},
        {"grid(1,1,9)",          {1,1,9},   {1,1,1},   0,   0,0,0, false},
        {"grid(100000,1,1)",     {100000,1,1},{1,1,1}, 0,   0,0,0, false},
        {"dynsmem=2048",         {1,1,1},   {1,1,1},   2048,0,0,0, false},
        {"cluster(4,1,1)",       {8,1,1},   {1,1,1},   0,   4,1,1, false},
        {"cluster(2,2,1)",       {8,1,1},   {1,1,1},   0,   2,2,1, false},
        {"coop grid(32,1,1)",    {32,1,1},  {1,1,1},   0,   0,0,0, true},
    };
    const int NC = sizeof(cfgs)/sizeof(cfgs[0]);
    uint32_t cols[NC][NW];

    for (int c = 0; c < NC; ++c) {
        cudaMemset(d, 0, NW*sizeof(uint32_t));
        Cfg& cf = cfgs[c];
        cudaError_t e;
        if (cf.coop) {
            void* args[] = {&dummy, &wbase, &d};
            e = cudaLaunchCooperativeKernel((void*)probe, cf.g, cf.b, args, cf.dyn, 0);
        } else if (cf.cx) {
            cudaLaunchConfig_t lc = {};
            lc.gridDim = cf.g; lc.blockDim = cf.b; lc.dynamicSmemBytes = cf.dyn;
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim = {(unsigned)cf.cx,(unsigned)cf.cy,(unsigned)cf.cz};
            lc.attrs = attr; lc.numAttrs = 1;
            e = cudaLaunchKernelEx(&lc, probe, dummy, wbase, d);
        } else {
            probe<<<cf.g, cf.b, cf.dyn>>>(dummy, wbase, d);
            e = cudaGetLastError();
        }
        if (!e) e = cudaDeviceSynchronize();
        if (e) { printf("%-20s LAUNCH ERR: %s\n", cf.tag, cudaGetErrorString(e)); 
                 memset(cols[c],0,sizeof(cols[c])); continue; }
        cudaMemcpy(cols[c], d, NW*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    // Print header
    printf("%-6s", "off");
    for (int c=0;c<NC;c++) printf(" %10s", cfgs[c].tag);
    printf("\n");
    // Print rows where any column is nonzero OR differs from base
    for (int i=0;i<NW;i++) {
        bool show=false;
        for (int c=0;c<NC;c++) if (cols[c][i] || cols[c][i]!=cols[0][i]) show=true;
        if (!show) continue;
        printf("0x%03x", i*4);
        for (int c=0;c<NC;c++) printf(" %10x", cols[c][i]);
        printf("\n");
    }
    return 0;
}
