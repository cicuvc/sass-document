#include <cstdint>
// warp-divergent + convergence-heavy code to try to elicit BRA.DIV/.CONV
__global__ void warpdiv(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = in[i], acc = 0;
    #pragma unroll 1
    for (int k = 0; k < n; ++k) {
        if (v & (1 << (k & 31))) {
            acc += __shfl_xor_sync(0xffffffff, v, 1);
            if (acc > 1000) { acc = -1; break; }
        } else {
            acc -= __shfl_down_sync(0xffffffff, v, 2);
        }
        v = __funnelshift_l(v, acc, 1);
    }
    // big straight-line body to push some branch targets far apart
    #pragma unroll
    for (int j = 0; j < 64; ++j) acc = acc * 1664525 + 1013904223 + in[(i + j) % n];
    out[i] = acc;
}
