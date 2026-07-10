#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Is L2 a small monolithic (1-2 slice) OoO atomic backend, or many slices?
// Peak aggregate atomic throughput to DISTINCT addresses (no same-addr contention).
// Sweep active warps; find where aggregate throughput saturates and read the
// ceiling. A 1-2 slice OoO backend saturates at a modest ceiling.
#define ITER 8192
#define GAP_WORDS 4096   // 16 KB apart -> distinct lines/hashed

__global__ void atk(uint32_t* buf){
    uint32_t* p = buf + (size_t)(blockIdx.x*blockDim.x + threadIdx.x)*GAP_WORDS;
    #pragma unroll 1
    for(int i=0;i<ITER;i++) atomicAdd(p, 1u);   // each lane a UNIQUE address
}
int main(){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    printf("dev=%s SMs=%d\n",pr.name,pr.multiProcessorCount);
    size_t maxlanes=(size_t)1024*256;
    uint32_t* buf; cudaMalloc(&buf,maxlanes*GAP_WORDS*4); cudaMemset(buf,0,maxlanes*GAP_WORDS*4);
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
    printf("%-8s %-10s %-12s\n","lanes","Matom/s","per-lane");
    for(int total : {32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072}){
        int threads=256, blocks=(total+threads-1)/threads;
        long long n=(long long)blocks*threads*ITER;
        atk<<<blocks,threads>>>(buf); cudaDeviceSynchronize();
        cudaEventRecord(a); atk<<<blocks,threads>>>(buf); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms,a,b);
        printf("%-8d %-10.0f %-12.3f\n",blocks*threads, n/(ms*1e6), n/(ms*1e6)/(blocks*threads));
    }
    return 0;
}
