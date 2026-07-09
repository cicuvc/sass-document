#include <cstdio>
#include <cstdint>

struct Big { uint32_t a[4]; };
#define NW 132

__device__ void dump(const Big& b, int wbase, uint32_t* out) {
    #pragma unroll 1
    for (int i = 0; i < NW; ++i) out[i] = b.a[wbase + i];
}

// minimal: few regs, no local
__global__ void kMin(__grid_constant__ const Big b, int wbase, uint32_t *out) {
    if (threadIdx.x==0) dump(b, wbase, out);
}
// large per-thread local memory (dynamic index -> real local frame)
template<int NLOC>
__global__ void kLocal(__grid_constant__ const Big b, int wbase, uint32_t *out, int seed) {
    volatile int loc[NLOC];
    #pragma unroll 1
    for (int i=0;i<NLOC;i++) loc[(i*seed)&(NLOC-1)] = i+seed;
    if (threadIdx.x==0) { dump(b, wbase, out); out[NW-1] += loc[(seed)&(NLOC-1)]&0; }
}
// high register pressure, capped by launch_bounds
__global__ void __launch_bounds__(128,1)
kReg(__grid_constant__ const Big b, int wbase, uint32_t *out, int s) {
    float r[32];
    #pragma unroll
    for (int i=0;i<32;i++) r[i]=s*0.1f+i;
    #pragma unroll
    for (int i=0;i<32;i++) r[i]=r[i]*r[(i+1)&31]+r[(i+2)&31];
    float acc=0;
    #pragma unroll
    for (int i=0;i<32;i++) acc+=r[i];
    if (threadIdx.x==0){ dump(b,wbase,out); if(acc==-1.f) out[0]+=1; }
}

template<class F>
void run(const char* tag, F kern, uint32_t* d, uint32_t* col, Big dummy, int wbase, int nargs) {
    cudaMemset(d,0,NW*4);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*)kern);
    printf("%-12s numRegs=%-3d localB=%-6zu sharedB=%-5zu constB=%-4zu maxTPB=%-4d ptx=%d bin=%d\n",
        tag, fa.numRegs, (size_t)fa.localSizeBytes, (size_t)fa.sharedSizeBytes,
        (size_t)fa.constSizeBytes, fa.maxThreadsPerBlock, fa.ptxVersion, fa.binaryVersion);
    cudaMemcpy(col, d, 0, cudaMemcpyDeviceToHost); // noop keep types
}

int main(){
    uint32_t *d; cudaMalloc(&d, NW*4);
    Big dummy{}; int wb=(0x00-0x210)/4;
    uint32_t cols[6][NW]; const char* tags[6];
    int n=0;
    auto go=[&](const char* tag, auto kern, auto launch){
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa,(const void*)kern);
        cudaMemset(d,0,NW*4);
        launch();
        cudaError_t e=cudaDeviceSynchronize();
        if(e){printf("%-14s ERR %s\n",tag,cudaGetErrorString(e));return;}
        cudaMemcpy(cols[n],d,NW*4,cudaMemcpyDeviceToHost);
        tags[n]=tag;
        printf("%-14s numRegs=%-3d localB=%-7zu sharedB=%-6zu constB=%-5zu maxTPB=%d\n",
            tag,fa.numRegs,(size_t)fa.localSizeBytes,(size_t)fa.sharedSizeBytes,
            (size_t)fa.constSizeBytes,fa.maxThreadsPerBlock);
        n++;
    };
    go("min", kMin, [&]{ kMin<<<1,32>>>(dummy,wb,d); });
    go("local256", kLocal<256>, [&]{ kLocal<256><<<1,32>>>(dummy,wb,d,3); });
    go("local8192", kLocal<8192>, [&]{ kLocal<8192><<<1,32>>>(dummy,wb,d,3); });
    go("reg32", kReg, [&]{ kReg<<<1,32>>>(dummy,wb,d,3); });

    printf("\n%-6s","off");
    for(int c=0;c<n;c++) printf(" %12s",tags[c]);
    printf("\n");
    for(int i=0;i<NW;i++){
        bool show=false; for(int c=0;c<n;c++) if(cols[c][i]!=cols[0][i]) show=true;
        if(!show) continue;
        printf("0x%03x",i*4);
        for(int c=0;c<n;c++) printf(" %12x",cols[c][i]);
        printf("\n");
    }
    return 0;
}
