#include <cstdint>
// No fence at all. Bad store, then a second store (patch target), back to back.
extern "C" __global__ void victim(uint32_t* good, uint32_t v){
    uint32_t* bad=(uint32_t*)0xdead0000deadbe00ULL;
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(bad),"r"(v):"memory");
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(v):"memory"); // patch->illegal
}
