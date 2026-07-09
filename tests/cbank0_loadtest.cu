#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };
#define NW 132

__global__ void probe(__grid_constant__ const Big b, int wbase, uint32_t *out)
{
    if (threadIdx.x) return;
    #pragma unroll 1
    for (int i = 0; i < NW; ++i) out[i] = b.a[wbase + i];
}

// heavy FMA burn to drive up power / perturb clock
__global__ void burn(float *sink, int iters)
{
    float x = threadIdx.x * 1e-3f + 1.0f, y = 0.9999f;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) { x = x*y + 0.5f; y = y*x + 0.5f; }
    if (x == -1.0f) sink[threadIdx.x] = x + y;
}

int main()
{
    uint32_t *d; cudaMalloc(&d, NW*sizeof(uint32_t));
    float *sink; cudaMalloc(&sink, 1024*sizeof(float));
    Big dummy{};
    int wbase = (0x00-0x210)/4;

    auto dump = [&](uint32_t* out){
        probe<<<1,1>>>(dummy, wbase, d);
        cudaDeviceSynchronize();
        cudaMemcpy(out, d, NW*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    };

    uint32_t idle[NW]; dump(idle);

    // sustained load on a separate stream
    cudaStream_t s; cudaStreamCreate(&s);
    for (int k=0;k<400;k++) burn<<<114*4, 256, 0, s>>>(sink, 200000);

    // dump repeatedly while the burn queue drains
    uint32_t load[NW]; bool changed=false; int ndumps=0;
    while (cudaStreamQuery(s) == cudaErrorNotReady) {
        dump(load); ndumps++;
        for (int i=0;i<NW;i++) if (load[i]!=idle[i]) {
            printf("SLOT CHANGED under load: 0x%03x  idle=0x%08x load=0x%08x\n",
                   i*4, idle[i], load[i]); changed=true;
        }
    }
    cudaStreamSynchronize(s);
    printf("dumps during load = %d; any slot changed (excl. counters/ptrs)? %s\n",
           ndumps, changed ? "YES (see above)" : "NO");
    return 0;
}
