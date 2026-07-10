#include <cstdint>
// bad store LAST, after the fence. Patching ERRBAR/CGAERRBAR->illegal puts the
// illegal instruction BEFORE the faulting store in execution order.
extern "C" __global__ void victim(uint32_t* good, uint32_t v){
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(v):"memory");   // valid
    __threadfence();                                                                  // MEMBAR ERRBAR CGAERRBAR CCTL
    uint32_t* bad=(uint32_t*)0xdead0000deadbe00ULL;
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(bad),"r"(v):"memory");     // faults, LAST
}
