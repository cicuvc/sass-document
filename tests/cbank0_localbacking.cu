#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };
#define NW 132

template<int NLOC>
__global__ void k(__grid_constant__ const Big b, int wbase, uint32_t *out, int seed) {
    volatile int loc[NLOC];
    #pragma unroll 1
    for (int i=0;i<NLOC;i++) loc[(i*seed)&(NLOC-1)] = i+seed;
    if (blockIdx.x==0 && threadIdx.x==0) {
        #pragma unroll 1
        for (int i=0;i<NW;i++) out[i] = b.a[wbase+i];
        out[NW-1] += loc[seed&(NLOC-1)]&0;
    }
}

static uint32_t A[NW], B[NW];
int main(){
    uint32_t *d; cudaMalloc(&d, NW*4);
    Big dummy{}; int wb=(0x00-0x210)/4;

    cudaMemset(d,0,NW*4);
    k<4><<<1,32>>>(dummy,wb,d,3);            // tiny local, tiny launch
    cudaDeviceSynchronize(); cudaMemcpy(A,d,NW*4,cudaMemcpyDeviceToHost);

    cudaMemset(d,0,NW*4);
    k<16384><<<228,1024>>>(dummy,wb,d,3);    // 64KB/thread local, full-device occupancy
    cudaError_t e=cudaDeviceSynchronize();
    if(e){printf("heavy launch ERR: %s\n",cudaGetErrorString(e));return 1;}
    cudaMemcpy(B,d,NW*4,cudaMemcpyDeviceToHost);

    size_t freeB,totB; cudaMemGetInfo(&freeB,&totB);
    printf("after heavy: free=%zu MiB\n", freeB>>20);
    printf("%-6s %14s %14s\n","off","tiny","heavy64K/228x1024");
    for(int i=0;i<NW;i++) if(A[i]!=B[i])
        printf("0x%03x %14x %14x\n", i*4, A[i], B[i]);
    return 0;
}
