__global__ void nested_break(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            int v = in[(i*n + a)*n + b];
            acc += v;
            if (v < 0) { acc = -1; goto done; }   // break to outer
            if (v == 7) break;                      // inner break
        }
        if (acc > 1000) break;
    }
done:
    out[i] = acc;
}
