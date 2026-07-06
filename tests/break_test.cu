// Varied loop-break patterns to exercise BREAK with different barriers/predicates.

// Case A: while-loop with data-dependent break (single barrier)
__global__ void kA(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0, k = 0;
    while (k < n) {
        int v = in[i*n + k];
        acc += v;
        if (v > 255) break;
        k++;
    }
    out[i] = acc;
}

// Case B: triple-nested with break to different levels (multiple barriers B0/B1/B2)
__global__ void kB(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            for (int c = 0; c < n; ++c) {
                int v = in[((i*n+a)*n+b)*n+c];
                acc += v;
                if (v == 1) break;            // break inner
                if (v == 2) goto mid;         // break to mid
                if (v == 3) goto out_all;     // break all
            }
        }
        mid:;
        if (acc > 500) break;
    }
out_all:
    out[i] = acc;
}
