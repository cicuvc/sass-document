// HMMA (tensor-core, warp-level mma.sync) pipeline test.
// Chain 1: dependent accumulation into the same C fragment (RAW HMMA->HMMA on
//          the accumulator) -> exposes how the ~28-cyc table latency is scheduled.
// Chain 2: independent MMAs (different accumulators) -> throughput / issue gap.
#include <cuda_fp16.h>
#include <cstdint>

// m16n8k16 f16.f16.f16.f16 : A={a0..a3}, B={b0,b1}, C/D={c0,c1}
#define MMA(d0,d1,a0,a1,a2,a3,b0,b1,c0,c1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n" \
    : "=r"(d0),"=r"(d1) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(c0),"r"(c1))

extern "C" __global__ void hmma_accum_chain(uint32_t* out, const uint32_t* in) {
    uint32_t a0=in[0],a1=in[1],a2=in[2],a3=in[3],b0=in[4],b1=in[5];
    uint32_t c0=in[6],c1=in[7];
#pragma unroll
    for (int i = 0; i < 32; i++) {
        MMA(c0,c1, a0,a1,a2,a3, b0,b1, c0,c1);   // C = A*B + C  (serial on C)
    }
    out[threadIdx.x*2+0]=c0; out[threadIdx.x*2+1]=c1;
}

extern "C" __global__ void hmma_indep(uint32_t* out, const uint32_t* in) {
    uint32_t a0=in[0],a1=in[1],a2=in[2],a3=in[3],b0=in[4],b1=in[5];
    uint32_t acc[8];
#pragma unroll
    for (int k = 0; k < 4; k++) { acc[2*k]=in[6+2*k]; acc[2*k+1]=in[7+2*k]; }
#pragma unroll
    for (int k = 0; k < 4; k++) {                 // 4 independent MMAs
        MMA(acc[2*k],acc[2*k+1], a0,a1,a2,a3, b0,b1, acc[2*k],acc[2*k+1]);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) out[threadIdx.x*8+k]=acc[k];
}
