// F2I (float->int convert) pipeline test.
// F2I is mio_pipe / VQ_MUFU / DECOUPLED_RD_WR_SCBD -> variable latency: it sets a
// write scoreboard, and a consumer of its result waits via req_bit_set.
// Cover src fmts (F32/F64/F16), dst fmts (S32/U32/S64), rounding, and a chain.
#include <cuda_fp16.h>
#include <cstdint>

extern "C" __global__ void f2i_variants(int* o, unsigned* uo, long long* lo,
                                         const float* f, const double* d,
                                         const __half* h) {
    int t = threadIdx.x;
    o[t]   = (int)f[t];             // F2I.F32.S32 (round toward zero)
    uo[t]  = (unsigned)f[t];        // F2I.F32.U32
    lo[t]  = (long long)d[t];       // F2I.F64.S64
    o[t+1] = (int)__half2float(h[t]);// F2I.F16.S32 (or via F16)
    o[t+2] = __float2int_rn(f[t]);  // F2I .RN
    o[t+3] = __float2int_rd(f[t]);  // F2I .RM (floor)
    o[t+4] = __float2int_ru(f[t]);  // F2I .RP (ceil)
}

// dependent: F2I result feeds an integer op immediately (consumer waits on the
// F2I write scoreboard) -> exposes the variable-latency handling.
extern "C" __global__ void f2i_dep(int* o, const float* f) {
    int t = threadIdx.x;
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int c = (int)f[t + i];      // F2I
        acc += c;                   // IADD3 consuming the F2I result (RAW)
    }
    o[t] = acc;
}
