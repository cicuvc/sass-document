#include <cstdint>

// Probe how an acquire LOAD orders subsequent loads.
// Question: does acquire gate ALL later loads (fence-like), or only enforce
// that the acquire load's result is consumed before dependents (scoreboard)?

// Case A: acquire load, then INDEPENDENT loads (no data dep on acquire result).
extern "C" __global__ void acq_then_indep(uint32_t *p, uint32_t *q, uint32_t *o){
    uint32_t a, b, c;
    asm volatile("ld.acquire.cta.global.u32 %0,[%1];":"=r"(a):"l"(p):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(b):"l"(q+0):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(c):"l"(q+1):"memory");
    o[0]=a; o[1]=b; o[2]=c;
}

// Case B: acquire load, then DEPENDENT load (address derived from acquire value).
extern "C" __global__ void acq_then_dep(uint32_t *p, uint32_t *base, uint32_t *o){
    uint32_t a, b;
    asm volatile("ld.acquire.cta.global.u32 %0,[%1];":"=r"(a):"l"(p):"memory");
    uint32_t *addr = base + (a & 15);
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(b):"l"(addr):"memory");
    o[0]=a; o[1]=b;
}

// Case C: relaxed load (baseline), then independent loads.
extern "C" __global__ void rlx_then_indep(uint32_t *p, uint32_t *q, uint32_t *o){
    uint32_t a, b, c;
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(a):"l"(p):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(b):"l"(q+0):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(c):"l"(q+1):"memory");
    o[0]=a; o[1]=b; o[2]=c;
}

// Case D: acquire load, then a relaxed STORE (independent) — does acquire gate stores?
extern "C" __global__ void acq_then_store(uint32_t *p, uint32_t *q, uint32_t *o){
    uint32_t a;
    asm volatile("ld.acquire.cta.global.u32 %0,[%1];":"=r"(a):"l"(p):"memory");
    asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(q),"r"(123):"memory");
    o[0]=a;
}
