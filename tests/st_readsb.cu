#include <cstdint>
// Force the store-data register to be overwritten right after the store, so the
// store must protect its source read with a READ scoreboard (src_rel_sb) or a
// stall. Inspect whether STG sets rd_sb!=7 and how the overwriter is gated.
extern "C" __global__ void st_reuse_g(uint32_t* p, uint32_t din, uint32_t* o){
    uint32_t d=din;
    asm volatile("st.relaxed.cta.global.u32 [%1], %0;\n\t"
                 "add.u32 %0, %0, 0x1234;\n\t"   // WAR on store-data reg d
                 : "+r"(d) : "l"(p) : "memory");
    o[0]=d;
}
extern "C" __global__ void st_reuse_s(uint32_t din, uint32_t* o){
    __shared__ uint32_t s[4];
    uint32_t a=(uint32_t)__cvta_generic_to_shared(&s[0]); uint32_t d=din;
    asm volatile("st.relaxed.cta.shared.u32 [%1], %0;\n\t"
                 "add.u32 %0, %0, 0x1234;\n\t"
                 : "+r"(d) : "r"(a) : "memory");
    o[0]=d;
}
// Also: address register reuse (does the store hold the ADDRESS reg late too?)
extern "C" __global__ void st_addr_reuse_g(uint32_t* p, uint32_t d, uint32_t* o){
    uint64_t pp=(uint64_t)p;
    asm volatile("st.relaxed.cta.global.u32 [%0], %2;\n\t"
                 "add.u64 %0, %0, 0x100;\n\t"    // WAR on address reg
                 : "+l"(pp) : "l"(pp), "r"(d) : "memory");
    o[0]=(uint32_t)pp;
}
