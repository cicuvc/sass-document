#include <cstdint>
// Victim: store to a wild (faulting) address, then a gpu fence (MEMBAR + ERRBAR
// + CGAERRBAR + CCTL.IVALL), then a normal store. We patch ERRBAR-slot vs
// CGAERRBAR-slot to an illegal instruction and observe which fault the runtime
// reports.
extern "C" __global__ void victim(uint32_t* good, uint32_t v){
    uint32_t* bad = (uint32_t*)0xdead0000deadbe00ULL;   // wild, unmapped
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(bad),"r"(v):"memory");
    __threadfence();                                    // MEMBAR.SC.GPU+ERRBAR+CGAERRBAR+CCTL
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;"::"l"(good),"r"(v+1):"memory");
}
