#include <cstdint>
// Acquire-folding probe. If the acquire NOP/CCTL can fold, it should disappear
// when the next instruction already naturally waits on the acquire load's
// scoreboard (data dependency). Patterns:
//   A) acquire then USE the loaded value immediately → natural dep (should fold)
//   B) acquire then independent work (no dep on loaded value) → must NOT fold
//   C) acquire then another load (relaxed) → must NOT fold (ordering matters)
//   D) acquire then atom.relaxed → must NOT fold

__device__ __forceinline__ uint32_t ld_acq_cta(uint32_t*p){uint32_t v;asm volatile("ld.acquire.cta.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ uint32_t ld_acq_gpu(uint32_t*p){uint32_t v;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void st_rlx(uint32_t*p,uint32_t v){asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}

// A) dependency: acquire then immediately USE → IADD uses acquire result
extern "C" __global__ void acq_then_use_cta(uint32_t*p,uint32_t*o){uint32_t a=ld_acq_cta(p); uint32_t b=a+1; o[0]=b;}
extern "C" __global__ void acq_then_use_gpu(uint32_t*p,uint32_t*o){uint32_t a=ld_acq_gpu(p); uint32_t b=a+1; o[0]=b;}
// B) no dependency: acquire then independent operation (different reg)
extern "C" __global__ void acq_then_indep_cta(uint32_t*p,uint32_t q,uint32_t*o){uint32_t a=ld_acq_cta(p); uint32_t b=q+a; o[0]=b;}
// C) acquire then relaxed load (ordering must hold)
extern "C" __global__ void acq_then_load_cta(uint32_t*p,uint32_t*q,uint32_t*o){uint32_t a=ld_acq_cta(p); uint32_t b; asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(b):"l"(q):"memory"); o[0]=a+b;}
// D) acquire then atom.relaxed (does NOP disappear? atom has its own scoreboard)
extern "C" __global__ void acq_then_atom_cta(uint32_t*p,uint32_t*q,uint32_t*o){
    uint32_t a=ld_acq_cta(p); uint32_t r; asm volatile("atom.relaxed.cta.global.add.u32 %0,[%1],%2;":"=r"(r):"l"(q),"r"(a):"memory"); o[0]=r;}
// E) acquire + fence.gpu (do the two CCTL.IVALLs merge?)
extern "C" __global__ void acq_then_fence_gpu(uint32_t*p,uint32_t*o){uint32_t a=ld_acq_gpu(p); __threadfence(); o[0]=a;}
