// Force local memory usage: a local array with dynamic (runtime) indexing
// that cannot be promoted to registers, plus a store to it.
__global__ void local_store(int *out, int n, int idx)
{
    int buf[64];
    // dynamic index store -> must live in local memory
    #pragma unroll 1
    for (int i = 0; i < n; ++i) {
        buf[(i * idx) & 63] = i * 7 + idx;
    }
    // read back with a dynamic index so buf can't be optimized away
    out[threadIdx.x] = buf[(n * idx) & 63];
}
