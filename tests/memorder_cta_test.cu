#include <cstdint>

// One kernel per (space x order x dir). Named so cuobjdump labels each block.
// .cta scope, general proxy, strong ops.

#define GLD(name, q) \
extern "C" __global__ void name(uint32_t*p,uint32_t*o){uint32_t v; \
  asm volatile(q " %0,[%1];":"=r"(v):"l"(p):"memory"); *o=v;}
#define GST(name, q) \
extern "C" __global__ void name(uint32_t*p,uint32_t v){ \
  asm volatile(q " [%0],%1;"::"l"(p),"r"(v):"memory");}
#define SLD(name, q) \
extern "C" __global__ void name(uint32_t*p,uint32_t*o){uint32_t v; \
  uint32_t a=(uint32_t)__cvta_generic_to_shared(p); \
  asm volatile(q " %0,[%1];":"=r"(v):"r"(a):"memory"); *o=v;}
#define SST(name, q) \
extern "C" __global__ void name(uint32_t*p,uint32_t v){ \
  uint32_t a=(uint32_t)__cvta_generic_to_shared(p); \
  asm volatile(q " [%0],%1;"::"r"(a),"r"(v):"memory");}

// global loads
GLD(g_ld_weak,     "ld.global.u32")
GLD(g_ld_relaxed,  "ld.relaxed.cta.global.u32")
GLD(g_ld_acquire,  "ld.acquire.cta.global.u32")
GLD(g_ld_volatile, "ld.volatile.global.u32")
GLD(g_ld_mmio,     "ld.mmio.relaxed.sys.global.u32")
// global stores
GST(g_st_weak,     "st.global.u32")
GST(g_st_relaxed,  "st.relaxed.cta.global.u32")
GST(g_st_release,  "st.release.cta.global.u32")
GST(g_st_volatile, "st.volatile.global.u32")
GST(g_st_mmio,     "st.mmio.relaxed.sys.global.u32")
// shared loads
SLD(s_ld_weak,     "ld.shared.u32")
SLD(s_ld_relaxed,  "ld.relaxed.cta.shared.u32")
SLD(s_ld_acquire,  "ld.acquire.cta.shared.u32")
SLD(s_ld_volatile, "ld.volatile.shared.u32")
// shared stores
SST(s_st_weak,     "st.shared.u32")
SST(s_st_relaxed,  "st.relaxed.cta.shared.u32")
SST(s_st_release,  "st.release.cta.shared.u32")
SST(s_st_volatile, "st.volatile.shared.u32")
