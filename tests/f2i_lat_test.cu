// F2I latency vs sequence test.
// F2I is scoreboard-tracked (VQ_MUFU), so its true latency is signalled at
// runtime, not a fixed stall. We probe it two ways:
//   (1) strict dependent chain: each F2I depends on the previous result via an
//       I2F round-trip -> scoreboard serialises them; measure issue spacing.
//   (2) many independent F2I -> MIO/MUFU issue throughput.
#include <cstdint>

// (1) dependent recirculation: x_{i+1} = (int)((float)x_i * 1.5f + 1.0f)
extern "C" __global__ void f2i_dep_chain(int* o, int start) {
    int x = start;
#pragma unroll
    for (int i = 0; i < 24; i++) {
        float f = (float)x;          // I2F  (scoreboard)
        f = f * 1.5f + 1.0f;         // FFMA (fixed latency)
        x = (int)f;                  // F2I  (scoreboard) -> feeds next I2F
    }
    o[threadIdx.x] = x;
}

// (2) independent: 16 unrelated floats each converted once, all live
extern "C" __global__ void f2i_indep(int* o, const float* f) {
    int t = threadIdx.x;
    int r[16];
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = (int)f[t + i];   // independent F2I
    int s = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) s ^= r[i];              // consume all (forces them live)
    o[t] = s;
}

// (3) pure back-to-back F2I with independent sources but same-latency, no float
//     ALU in between -> tightest F2I issue cadence
extern "C" __global__ void f2i_tput(int* o, const float* f) {
    int t = threadIdx.x;
    int a=(int)f[t],   b=(int)f[t+1], c=(int)f[t+2], d=(int)f[t+3];
    int e=(int)f[t+4], g=(int)f[t+5], h=(int)f[t+6], k=(int)f[t+7];
    o[t] = a+b+c+d+e+g+h+k;
}
