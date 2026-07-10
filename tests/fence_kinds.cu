#include <cstdint>
extern "C" __global__ void f_acqrel_cta(uint32_t*o){asm volatile("fence.acq_rel.cta;":::"memory");*o=1;}
extern "C" __global__ void f_acqrel_gpu(uint32_t*o){asm volatile("fence.acq_rel.gpu;":::"memory");*o=1;}
extern "C" __global__ void f_acqrel_sys(uint32_t*o){asm volatile("fence.acq_rel.sys;":::"memory");*o=1;}
extern "C" __global__ void f_sc_cta(uint32_t*o){asm volatile("fence.sc.cta;":::"memory");*o=1;}
extern "C" __global__ void f_sc_gpu(uint32_t*o){asm volatile("fence.sc.gpu;":::"memory");*o=1;}
extern "C" __global__ void f_sc_sys(uint32_t*o){asm volatile("fence.sc.sys;":::"memory");*o=1;}
