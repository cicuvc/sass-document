#include <cstdint>
// RO async store, then fence (MEMBAR ERRBAR CGAERRBAR CCTL), then good store.
extern "C" __global__ void victim(uint32_t* bad, uint32_t* good, uint32_t v){
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(bad),"r"(v):"memory");   // async RO fault
    __threadfence();
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(v):"memory");
}
