#include <cstdint>
// atom/red order-preserving SASS patterns. Focus: MEMBAR / CCTL.IVALL / ERRBAR /
// scoreboard-stall around ATOMG/REDG/ATOMS/REDS — NOT the encoding.
// PTX order: atom{.sem}{.scope}{.space}.op.type

#define ATOM(name,q) extern "C" __global__ void name(uint32_t*p,uint32_t v,uint32_t*o){ \
  uint32_t r; asm volatile(q " %0,[%1],%2;":"=r"(r):"l"(p),"r"(v):"memory"); *o=r; }
#define RED(name,q) extern "C" __global__ void name(uint32_t*p,uint32_t v){ \
  asm volatile(q " [%0],%1;"::"l"(p),"r"(v):"memory"); }
#define ATOMS(name,q) extern "C" __global__ void name(uint32_t*pp,uint32_t v,uint32_t*o){ \
  __shared__ uint32_t s[4]; uint32_t a=(uint32_t)__cvta_generic_to_shared(&s[0]); \
  uint32_t r; asm volatile(q " %0,[%1],%2;":"=r"(r):"r"(a),"r"(v):"memory"); *o=r+s[1]; }

// ---- global atom.add, all orders x scopes ----
ATOM(atom_relaxed_cta, "atom.relaxed.cta.global.add.u32")
ATOM(atom_acquire_cta, "atom.acquire.cta.global.add.u32")
ATOM(atom_release_cta, "atom.release.cta.global.add.u32")
ATOM(atom_acqrel_cta,  "atom.acq_rel.cta.global.add.u32")
ATOM(atom_relaxed_gpu, "atom.relaxed.gpu.global.add.u32")
ATOM(atom_acquire_gpu, "atom.acquire.gpu.global.add.u32")
ATOM(atom_release_gpu, "atom.release.gpu.global.add.u32")
ATOM(atom_acqrel_gpu,  "atom.acq_rel.gpu.global.add.u32")
ATOM(atom_acqrel_sys,  "atom.acq_rel.sys.global.add.u32")

// ---- global red.add (write-only), relaxed/release ----
RED(red_relaxed_cta, "red.relaxed.cta.global.add.u32")
RED(red_release_cta, "red.release.cta.global.add.u32")
RED(red_relaxed_gpu, "red.relaxed.gpu.global.add.u32")
RED(red_release_gpu, "red.release.gpu.global.add.u32")

// ---- shared atom.add ----
ATOMS(atoms_relaxed_cta, "atom.relaxed.cta.shared.add.u32")
ATOMS(atoms_acquire_cta, "atom.acquire.cta.shared.add.u32")
ATOMS(atoms_release_cta, "atom.release.cta.shared.add.u32")
ATOMS(atoms_acqrel_cta,  "atom.acq_rel.cta.shared.add.u32")

// ---- exch / cas patterns (common lock primitives) ----
ATOM(atom_exch_acq_gpu, "atom.acquire.gpu.global.exch.b32")
extern "C" __global__ void atom_cas_acqrel_gpu(uint32_t*p,uint32_t c,uint32_t v,uint32_t*o){
  uint32_t r; asm volatile("atom.acq_rel.gpu.global.cas.b32 %0,[%1],%2,%3;":"=r"(r):"l"(p),"r"(c),"r"(v):"memory"); *o=r; }
