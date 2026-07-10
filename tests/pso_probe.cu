#include <cstdint>
// Probe which orderings ptxas thinks it must enforce at .cta scope.
// If a barrier appears, that ordering is NOT free in the underlying model.

// (1) two relaxed stores to DIFFERENT addresses: barrier between? (store->store)
extern "C" __global__ void ss_relaxed(uint32_t* a, uint32_t* b){
    asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(a),"r"(1):"memory");
    asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(b),"r"(2):"memory");
}
// (2) two relaxed loads from DIFFERENT addresses: barrier between? (load->load)
extern "C" __global__ void ll_relaxed(uint32_t* a, uint32_t* b, uint32_t* o){
    uint32_t x,y;
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(x):"l"(a):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(y):"l"(b):"memory");
    o[0]=x; o[1]=y;
}
// (3) relaxed load then relaxed store, different addr (load->store)
extern "C" __global__ void ls_relaxed(uint32_t* a, uint32_t* b){
    uint32_t x;
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(x):"l"(a):"memory");
    asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(b),"r"(x):"memory");
}
// (4) MP producer with release: data(relaxed) ; flag(release)  -> where is the fence?
extern "C" __global__ void mp_prod_release(uint32_t* data, uint32_t* flag){
    asm volatile("st.relaxed.cta.global.u32 [%0],%1;"::"l"(data),"r"(42):"memory");
    asm volatile("st.release.cta.global.u32 [%0],%1;"::"l"(flag),"r"(1):"memory");
}
// (5) MP consumer with acquire: flag(acquire) ; data(relaxed) -> where is the barrier?
extern "C" __global__ void mp_cons_acquire(uint32_t* flag, uint32_t* data, uint32_t* o){
    uint32_t f,d;
    asm volatile("ld.acquire.cta.global.u32 %0,[%1];":"=r"(f):"l"(flag):"memory");
    asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(d):"l"(data):"memory");
    o[0]=f; o[1]=d;
}
