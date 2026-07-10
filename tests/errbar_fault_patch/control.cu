#include <cstdint>
extern "C" __global__ void victim(uint32_t* good, uint32_t v){
    __threadfence();   // MEMBAR+ERRBAR+CGAERRBAR+CCTL, no faulting store
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(v):"memory");
}
