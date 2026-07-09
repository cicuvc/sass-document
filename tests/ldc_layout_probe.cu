// A: different param layout + local memory
__global__ void kA(double *p, char c, long long q, float f)
{
    int buf[32];
    #pragma unroll 1
    for (int i = 0; i < (int)q; ++i) buf[(i * c) & 31] = i + (int)f;
    p[threadIdx.x] = buf[((int)q * c) & 31];
}

// B: two pointers, no local memory (just global load/store)
__global__ void kB(const float *a, float *b)
{
    b[threadIdx.x] = a[threadIdx.x] * 2.0f;
}

// C: many small params + local memory
__global__ void kC(int *out, int a, int b, int cc, int d, int e)
{
    int buf[48];
    #pragma unroll 1
    for (int i = 0; i < a; ++i) buf[(i * b) & 47] = i + cc + d + e;
    out[threadIdx.x] = buf[(a * b) & 47];
}
