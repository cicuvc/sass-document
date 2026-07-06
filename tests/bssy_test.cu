#include <cstdint>

// Case 1: data-dependent if/else -> divergent region, expect BSSY/BSYNC around it
__global__ void diverge_if(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    if (v > 0) {
        v = v * 3 + 1;
    } else {
        v = v * 7 - 5;
    }
    out[i] = v;
}

// Case 2: data-dependent loop with break -> expect BSSY + BREAK + BSYNC
__global__ void diverge_break(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    for (int k = 0; k < n; ++k) {
        int v = in[i * n + k];
        acc += v;
        if (v == 0) break;          // divergent break
    }
    out[i] = acc;
}

// Case 3: nested divergence -> nested convergence barriers (B0, B1)
__global__ void diverge_nested(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    if (v > 10) {
        if (v > 100) {
            v = v - 100;
        } else {
            v = v + 100;
        }
    }
    out[i] = v;
}

// Case 4: switch -> multiway divergence
__global__ void diverge_switch(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    switch (v & 3) {
        case 0: v += 1; break;
        case 1: v *= 2; break;
        case 2: v -= 3; break;
        default: v = -v; break;
    }
    out[i] = v;
}
