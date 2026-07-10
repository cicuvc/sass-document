#include <cstdint>

// .gpu / .sys scope acquire/release, global (L1/L2). Compare to .cta.
// Pattern: acquire load then independent load (to expose any fence/barrier),
// and release store then nothing.

#define ACQ_INDEP(name, q) \
extern "C" __global__ void name(uint32_t *p, uint32_t *r, uint32_t *o){ \
  uint32_t a,b; \
  asm volatile(q " %0,[%1];":"=r"(a):"l"(p):"memory"); \
  asm volatile("ld.relaxed.cta.global.u32 %0,[%1];":"=r"(b):"l"(r):"memory"); \
  o[0]=a; o[1]=b; }

#define REL(name, q) \
extern "C" __global__ void name(uint32_t *p, uint32_t v){ \
  asm volatile(q " [%0],%1;"::"l"(p),"r"(v):"memory"); }

#define RLX_LD(name, q) \
extern "C" __global__ void name(uint32_t *p, uint32_t *o){ uint32_t a; \
  asm volatile(q " %0,[%1];":"=r"(a):"l"(p):"memory"); o[0]=a; }

// acquire loads at each scope, followed by an independent relaxed load
ACQ_INDEP(acq_cta, "ld.acquire.cta.global.u32")
ACQ_INDEP(acq_gpu, "ld.acquire.gpu.global.u32")
ACQ_INDEP(acq_sys, "ld.acquire.sys.global.u32")

// plain relaxed loads at each scope (baseline opcode)
RLX_LD(rlx_gpu, "ld.relaxed.gpu.global.u32")
RLX_LD(rlx_sys, "ld.relaxed.sys.global.u32")

// release stores at each scope
REL(rel_cta, "st.release.cta.global.u32")
REL(rel_gpu, "st.release.gpu.global.u32")
REL(rel_sys, "st.release.sys.global.u32")
