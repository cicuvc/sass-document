// DEPBAR test: cp.async (LDGSTS) groups + cp.async.wait_group -> DEPBAR.LE / LDGDEPBAR.
// DEPBAR.LE SBn, cnt waits until scoreboard SBn's outstanding count <= cnt.
#include <cstdint>

extern "C" __global__ void depbar_cpasync(float* out, const float* in) {
    __shared__ float smem[512];
    int t = threadIdx.x;
    unsigned s0 = __cvta_generic_to_shared(&smem[t]);
    unsigned s1 = __cvta_generic_to_shared(&smem[t + 256]);

    // group 0: two async copies
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s0), "l"(in + t));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s1), "l"(in + t + 256));
    asm volatile("cp.async.commit_group;\n");            // -> LDGDEPBAR (commit)

    // group 1: two more
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s0), "l"(in + t + 512));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s1), "l"(in + t + 768));
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 1;\n");            // wait until <=1 group left -> DEPBAR.LE ,0x1
    __syncthreads();
    float a = smem[t];

    asm volatile("cp.async.wait_group 0;\n");            // wait all -> DEPBAR.LE ,0x0
    __syncthreads();
    out[t] = a + smem[t + 256];
}

extern "C" __global__ void depbar_waitall(float* out, const float* in) {
    __shared__ float smem[256];
    int t = threadIdx.x;
    unsigned s0 = __cvta_generic_to_shared(&smem[t]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s0), "l"(in + t));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");                // drain everything
    __syncthreads();
    out[t] = smem[t];
}

// Proof that the scoreboard counts GROUPS not copies: groups of 4/1/2 copies,
// wait_group 2/1/0 -> DEPBAR.LE SB0, 0x2/0x1/0x0 (cnt = group count).
extern "C" __global__ void depbar_group_count(float* out, const float* in) {
    __shared__ float smem[2048];
    int t = threadIdx.x;
    unsigned s = __cvta_generic_to_shared(&smem[t]);
    for (int i = 0; i < 4; i++)                         // group A: 4 copies
        asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n"::"r"(s+1024*i),"l"(in+t+256*i));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n"::"r"(s+4096),"l"(in+t+1024)); // group B: 1
    asm volatile("cp.async.commit_group;\n");
    for (int i = 0; i < 2; i++)                         // group C: 2 copies
        asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n"::"r"(s+5120+1024*i),"l"(in+t+1280+256*i));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 2;\n"); __syncthreads(); float a=smem[t];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads(); a+=smem[t+256];
    asm volatile("cp.async.wait_group 0;\n"); __syncthreads();
    out[t]=a+smem[t+512];
}
