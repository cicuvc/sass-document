#include <cstdint>
#include <cuda_runtime.h>
// cp.reduce.async.bulk SASS. Common form: shared -> global .add, bulk_group.
extern "C" __global__ void tma_red_add_global(void* gdst, const uint32_t* in, uint32_t nbytes){
    extern __shared__ char smemb[]; uint32_t* smem=(uint32_t*)smemb;
    uint32_t smem_a=(uint32_t)__cvta_generic_to_shared(smem);
    smem[threadIdx.x]=in[threadIdx.x];
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0],[%1],%2;"
            ::"l"(gdst),"r"(smem_a),"r"(nbytes):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    }
}
// float add variant (different redop encoding?)
extern "C" __global__ void tma_red_add_f32(void* gdst, const float* in, uint32_t nbytes){
    extern __shared__ char smemb[]; float* smem=(float*)smemb;
    uint32_t smem_a=(uint32_t)__cvta_generic_to_shared(smem);
    smem[threadIdx.x]=in[threadIdx.x];
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0],[%1],%2;"
            ::"l"(gdst),"r"(smem_a),"r"(nbytes):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    }
}
// min variant
extern "C" __global__ void tma_red_min_s32(void* gdst, const int* in, uint32_t nbytes){
    extern __shared__ char smemb[]; int* smem=(int*)smemb;
    uint32_t smem_a=(uint32_t)__cvta_generic_to_shared(smem);
    smem[threadIdx.x]=in[threadIdx.x];
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%0],[%1],%2;"
            ::"l"(gdst),"r"(smem_a),"r"(nbytes):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    }
}
