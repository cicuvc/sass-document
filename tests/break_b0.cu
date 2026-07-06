// Loop whose break must peel the loop's own (outer-most) barrier while
// branching forward past a tail that itself contains divergence.
__global__ void kC(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = in[i];
    for (int k = 0; k < n; ++k) {
        int v = in[i*n + k];
        if (v == 42) break;                 // early exit of THIS loop
        acc += v * v;
        if (acc & 1) acc ^= v;              // extra divergence in body -> inner barrier
        else         acc += (v >> 1);
    }
    if (acc > 0) acc = acc % 97;            // tail divergence after loop
    out[i] = acc;
}
