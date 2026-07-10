#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Does a pre-fault mmio.sys write to pinned host memory survive a kernel fault?
__device__ __forceinline__ void mark(volatile uint32_t* h, int i, uint32_t v){
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0],%1;"::"l"(h+i),"r"(v):"memory");
}
__global__ void k(volatile uint32_t* h, uint32_t* bad, int do_fault){
    mark(h,0,0xAA);                     // progress marker BEFORE fault
    __threadfence_system();
    if(do_fault){ asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(bad),"r"(1):"memory"); }
    __threadfence_system();
    mark(h,1,0xBB);                     // progress marker AFTER (should be absent if killed)
}
int main(int argc,char**argv){
    int fault = argc>1 && atoi(argv[1]);
    uint32_t* hbuf; cudaHostAlloc(&hbuf,64,cudaHostAllocMapped);
    hbuf[0]=hbuf[1]=0;
    uint32_t* d_bad=(uint32_t*)0xdead0000deadbe00ULL; // unmapped -> sync fault (baseline)
    volatile uint32_t* dh; cudaHostGetDevicePointer((void**)&dh,hbuf,0);
    k<<<1,1>>>(dh,d_bad,fault);
    cudaError_t e=cudaDeviceSynchronize();
    printf("fault=%d  sync=%s  host[0]=0x%x host[1]=0x%x\n",fault,cudaGetErrorString(e),hbuf[0],hbuf[1]);
    return 0;
}
