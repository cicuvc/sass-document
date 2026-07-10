#include <cstdint>
// Is ERRBAR/CGAERRBAR tied to release, or a general MEMBAR@gpu+ companion?
extern "C" __global__ void fence_block(uint32_t*o){ __threadfence_block(); *o=1; }   // membar.cta
extern "C" __global__ void fence_gpu(uint32_t*o){ __threadfence(); *o=1; }            // membar.gl (gpu)
extern "C" __global__ void fence_sys(uint32_t*o){ __threadfence_system(); *o=1; }     // membar.sys
// plain relaxed store at gpu (no fence) for contrast
extern "C" __global__ void st_rlx_gpu(uint32_t*p,uint32_t v){ asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory"); }
