// LDC encoding exploration — PTX ld.const without bank qualifier
// Explicit constant banks are deprecated since PTX 2.2

// Test 1: ld.const.u32 with register address (should give LDC with Ra)
__device__ __noinline__ int ldc_reg_addr(const int *cptr) {
    int r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.u32 %0, [%1];" : "=r"(r) : "l"(addr));
    return r;
}

// Test 2: ld.const.b32 — raw bit-size load
__device__ __noinline__ int ldc_b32(const int *cptr) {
    int r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.b32 %0, [%1];" : "=r"(r) : "l"(addr));
    return r;
}

// Test 3: ld.const.u64
__device__ __noinline__ long long ldc_u64(const long long *cptr) {
    long long r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.u64 %0, [%1];" : "=l"(r) : "l"(addr));
    return r;
}

// Test 4: ld.const.u8
__device__ __noinline__ unsigned int ldc_u8(const unsigned char *cptr) {
    unsigned int r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.u8 %0, [%1];" : "=r"(r) : "l"(addr));
    return r;
}

// Test 5: ld.const.s16
__device__ __noinline__ int ldc_s16(const short *cptr) {
    int r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.s16 %0, [%1];" : "=r"(r) : "l"(addr));
    return r;
}

// Test 6: ld.const.s8
__device__ __noinline__ int ldc_s8(const char *cptr) {
    int r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.s8 %0, [%1];" : "=r"(r) : "l"(addr));
    return r;
}

// Test 7: ld.const.f32
__device__ __noinline__ float ldc_f32(const float *cptr) {
    float r;
    unsigned long long addr = (unsigned long long)cptr;
    asm("ld.const.f32 %0, [%1];" : "=f"(r) : "l"(addr));
    return r;
}

// Test 8: Predicated ld.const
__device__ __noinline__ int ldc_pred(const int *cptr, int cond) {
    int r = 0;
    unsigned long long addr = (unsigned long long)cptr;
    asm("{\n\t"
        ".reg .pred myp;\n\t"
        "setp.ne.s32 myp, %2, 0;\n\t"
        "@myp ld.const.u32 %0, [%1];\n\t"
        "}" : "+r"(r) : "l"(addr), "r"(cond));
    return r;
}

// Test 9: ld.const with C++ __constant__ — see what Compiler emits
// Use indexed access into constant memory to check if LDC or ULDC
__constant__ int test_cdata[256];
__device__ __noinline__ int ldc_from_constant_mem(int idx) {
    return test_cdata[idx];
}

extern "C" __global__ void ldc_ptx_kernel(
    int *out,
    const int *cptr0)
{
    out[0] = ldc_reg_addr(cptr0);
    out[1] = ldc_b32(cptr0);
    out[2] = (int)ldc_u64((const long long*)cptr0);
    out[3] = (int)ldc_u8((const unsigned char*)cptr0);
    out[4] = ldc_s16((const short*)cptr0);
    out[5] = ldc_s8((const char*)cptr0);
    out[6] = (int)ldc_f32((const float*)cptr0);
    out[7] = ldc_pred(cptr0, 1);
    out[8] = ldc_from_constant_mem(0);
}
