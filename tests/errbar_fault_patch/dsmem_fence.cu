#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__device__ __forceinline__ void mark(volatile uint32_t*h,int i,uint32_t v){
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0],%1;"::"l"(h+i),"r"(v):"memory");
}
// fmode: 0=no fence, 1=fence.cta(MEMBAR.SC.CTA only), 2=fence.gpu(+ERRBAR+CGAERRBAR)
extern "C" __global__ void __cluster_dims__(2,1,1) victim(volatile uint32_t* h, int bad_rank, int fmode){
    __shared__ uint32_t smem[64];
    unsigned rank = cg::this_cluster().block_rank();
    if(rank==0){
        uint32_t local=(uint32_t)__cvta_generic_to_shared(&smem[0]); uint32_t ds;
        asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(ds):"r"(local),"r"(bad_rank));
        asm volatile("st.shared::cluster.u32 [%0],%1;"::"r"(ds),"r"(0x1234):"memory");  // async fault
        mark(h,1,1);                       // passed store
        if(fmode==1) __threadfence_block();
        else if(fmode==2) __threadfence();
        mark(h,2,1);                       // passed fence (0 => fault collected at fence)
        for(int i=0;i<20;i++){ mark(h,3,i+1); }
        mark(h,4,0xD09E);
    }
    cg::this_cluster().sync();
}
int main(int argc,char**argv){
    int fmode=argc>1?atoi(argv[1]):2;
    uint32_t* hbuf; cudaHostAlloc(&hbuf,256,cudaHostAllocMapped); for(int i=0;i<64;i++)hbuf[i]=0;
    volatile uint32_t* dh; cudaHostGetDevicePointer((void**)&dh,hbuf,0);
    victim<<<2,1>>>(dh,7,fmode);
    cudaError_t e=cudaDeviceSynchronize();
    const char* fn[]={"none","fence.cta","fence.gpu"};
    printf("fmode=%s sync=%s | store[1]=%u fence[2]=%u loop[3]=%u DONE[4]=0x%x\n",
        fn[fmode],cudaGetErrorString(e),hbuf[1],hbuf[2],hbuf[3],hbuf[4]);
    return 0;
}
