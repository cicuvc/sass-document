// LDC encoding verification — sm_90
// Load from constant memory c[bank][offset]

__constant__ float cdata_f32[256];
__constant__ double cdata_f64[128];
__constant__ unsigned char cdata_u8[256];
__constant__ char cdata_s8[256];
__constant__ unsigned short cdata_u16[256];
__constant__ short cdata_s16[256];

// LDC.32 — reading kernel param (32-bit)
__device__ __noinline__ int ldc_kernel_param_i32(int a) {
    return a;
}

// LDC.64 — reading kernel param (64-bit)
__device__ __noinline__ long long ldc_kernel_param_i64(long long a) {
    return a;
}

// LDC.32 from __constant__ memory (indexed by Ra)
__device__ __noinline__ int ldc_const_f32(int idx) {
    return *((int*)&cdata_f32[idx]);
}

// LDC.64 from __constant__ memory
__device__ __noinline__ double ldc_const_f64(int idx) {
    return cdata_f64[idx];
}

// LDC.U8 from __constant__ memory
__device__ __noinline__ unsigned int ldc_const_u8(int idx) {
    return cdata_u8[idx];
}

// LDC.S8 from __constant__ memory
__device__ __noinline__ int ldc_const_s8(int idx) {
    return cdata_s8[idx];
}

// LDC.U16 from __constant__ memory
__device__ __noinline__ unsigned int ldc_const_u16(int idx) {
    return cdata_u16[idx];
}

// LDC.S16 from __constant__ memory
__device__ __noinline__ int ldc_const_s16(int idx) {
    return cdata_s16[idx];
}

// LDC with offset (bank + offset) — using a local struct with known alignment
struct alignas(16) pod { int x; int y; int z; int w; };

__device__ __noinline__ int ldc_const_struct_idx(int i) {
    const pod *p = (const pod*)cdata_f32;
    return p[i].z; // offset within struct
}

extern "C" __global__ void ldc_kernel(int *out,
                                      int a,
                                      long long b,
                                      int idx) {
    out[0] = ldc_kernel_param_i32(a);
    out[1] = (int)ldc_kernel_param_i64(b);
    out[2] = ldc_const_f32(idx);
    out[3] = (int)ldc_const_f64(idx);
    out[4] = (int)ldc_const_u8(idx);
    out[5] = ldc_const_s8(idx);
    out[6] = (int)ldc_const_u16(idx);
    out[7] = ldc_const_s16(idx);
    out[8] = ldc_const_struct_idx(idx);
}

// Uniform load test — use many kernel params to trigger ULDC via ptxas
struct big_params {
    int a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
};

__device__ __noinline__ int uldc_read_params(const big_params *p) {
    return p->a + p->b + p->c + p->d;
}

extern "C" __global__ void uldc_kernel(int *out, big_params params) {
    out[0] = uldc_read_params(&params);
}
