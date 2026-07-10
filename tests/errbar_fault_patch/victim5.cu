#include <cstdint>
// bad = a pointer passed by host (will be a mapped-but-read-only VMM region).
// good = normal writable buffer; its store is the patch target.
extern "C" __global__ void victim(uint32_t* bad, uint32_t* good, uint32_t v){
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(bad),"r"(v):"memory");   // RO -> protection fault
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(v):"memory");  // patch -> illegal
}
