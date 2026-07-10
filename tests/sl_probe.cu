#include <cstdint>
// store to A then load from independent B, all relaxed.cta. Does ptxas force
// the store to complete before the load (scoreboard/barrier), or are they free?
extern "C" __global__ void sl_indep(uint32_t* a, uint32_t* b, uint32_t* o){
    uint32_t v;
    asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(a),"r"(1):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(v):"l"(b):"memory");
    o[0]=v;
}
