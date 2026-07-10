#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Slice-count probe v2. Each thread hits address base+(tid%M)*STRIDE with M
// distinct addresses. Using the atomic RETURN value (accumulated + written out)
// prevents the compiler from combining the repeated atomics. Sweep M: aggregate
// throughput rises while M<N_slices (each new address ~ new slice) then saturates
// once all slices are busy -> the knee ≈ number of L2 slices.
#define ITER 256
#define STRIDE_WORDS 32   // 128 B = one L2 line, distinct lines -> hashed across slices

__global__ void atk(uint32_t* buf, int M, uint32_t* sink){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t* p=&buf[(size_t)(tid%M)*STRIDE_WORDS];
    uint32_t acc=0;
    #pragma unroll 1
    for(int i=0;i<ITER;i++) acc += atomicAdd(p,1u);   // return used -> not combinable
    if(acc==0xffffffffu) sink[tid]=acc;               // keep acc live
}
int main(){
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    int blocks=pr.multiProcessorCount*8, threads=256;
    long long natom=(long long)blocks*threads*ITER;
    printf("SMs=%d, blocks=%d threads=%d\n",pr.multiProcessorCount,blocks,threads);
    size_t words=(size_t)1024*STRIDE_WORDS*64; // room for up to 64k distinct lines
    uint32_t* buf; cudaMalloc(&buf,words*4); cudaMemset(buf,0,words*4);
    uint32_t* sink; cudaMalloc(&sink,(size_t)blocks*threads*4);
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
    auto tput=[&](int M){ atk<<<blocks,threads>>>(buf,M,sink); cudaDeviceSynchronize();
        cudaEventRecord(a); atk<<<blocks,threads>>>(buf,M,sink); cudaEventRecord(b);
        cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); return natom/(ms*1e6); };
    double prev=0; printf("%-8s %-12s %-8s\n","M","Matom/s","dT");
    for(int M : {1,2,4,8,12,16,24,32,48,64,80,96,112,128,160,192,256,384,512,1024,2048,4096}){
        double T=tput(M); printf("%-8d %-12.1f %+.1f\n",M,T,T-prev); prev=T;
    }
    return 0;
}
