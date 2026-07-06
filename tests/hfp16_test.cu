// fp16-pipe forwarding test: purely serial dependent chains of HFMA2/HADD2/HMUL2.
// A serial chain has zero ILP, so ptxas must set the producer stall = the
// effective RAW latency -> reading usched_info gives minG directly.
//   fp16 TABLE_TRUE(GPR) FP16->FP16 = 5.  forwarding => stall 4, none => stall 5.
#include <cuda_fp16.h>

extern "C" __global__ void hfma2_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) acc = __hfma2(acc, b, a);   // acc = acc*b + a
    out[threadIdx.x] = acc;
}

extern "C" __global__ void hadd2_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) acc = __hadd2(acc, b);      // acc = acc + b
    out[threadIdx.x] = acc;
}

extern "C" __global__ void hmul2_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) acc = __hmul2(acc, b);      // acc = acc * b
    out[threadIdx.x] = acc;
}

// mixed hfma2 -> hadd2 alternating dependent chain
extern "C" __global__ void hmix_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        acc = __hfma2(acc, b, a);
        acc = __hadd2(acc, b);
    }
    out[threadIdx.x] = acc;
}

// pure fp16-pipe chains that ptxas won't fuse into HFMA2.MMA
extern "C" __global__ void hmax2_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) acc = __hmax2(acc, b);      // HMNMX2 chain
    out[threadIdx.x] = acc;
}
extern "C" __global__ void hmaxmul_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) { acc = __hmax2(acc, b); acc = __hmul2(acc, b); }
    out[threadIdx.x] = acc;
}

extern "C" __global__ void haddmax_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) { acc = __hadd2(acc, b); acc = __hmax2(acc, b); }
    out[threadIdx.x] = acc;
}
extern "C" __global__ void hmaxadd_chain(__half2* out, __half2 a, __half2 b) {
    __half2 acc = a;
#pragma unroll
    for (int i = 0; i < 64; i++) { acc = __hmax2(acc, b); acc = __hadd2(acc, b); }
    out[threadIdx.x] = acc;
}
