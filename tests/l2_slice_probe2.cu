#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// One warp (block) per distinct hot line. 32 lanes' same-address atomics coalesce
// -> each block drives ~one slice's atomic stream. Sweep #blocks (=#lines=#slices
// exercised): aggregate throughput scales until all slices busy, then saturates.
// Knee ≈ L2 slice count. Distinct lines are spread by a large stride + hashing.
#define ITER 4096
#define STRIDE_WORDS 4096   // 16 KB apart, distinct slices (via addr hash)
__global__ void atk(uint32_t* buf, uint32_t* sink){
    uint32_t* p=&buf[(size_t)blockIdx.x*STRIDE_WORDS];
    uint32_t acc=0;
    #pragma unroll 1
    for(int i=0;i<ITER;i++) acc+=atomicAdd(p,1u);
    if(acc==0xffffffffu) sink[blockIdx.x]=acc;
}
int main(){
    size_t words=(size_t)8192*STRIDE_WORDS; uint32_t*buf; cudaMalloc(&buf,words*4); cudaMemset(buf,0,words*4);
    uint32_t*sink; cudaMalloc(&sink,8192*4);
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
    auto tput=[&](int G){ long long n=(long long)G*32*ITER;
        atk<<<G,32>>>(buf,sink); cudaDeviceSynchronize();
        cudaEventRecord(a); atk<<<G,32>>>(buf,sink); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms,a,b); return n/(ms*1e6); };
    printf("%-8s %-12s\n","warps","Matom/s");
    for(int G : {1,2,4,8,16,24,32,40,48,56,64,80,96,128,160,192,256,384,512,768,1024}){
        printf("%-8d %-12.1f\n",G,tput(G));
    }
    return 0;
}
