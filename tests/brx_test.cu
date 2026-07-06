// BRX / BRXU are register-indirect branches used for compiler-built jump tables.
//
// They are NOT reachable from CUDA C on sm_90 with CUDA 13.1:
//   - address-of-label / computed goto is rejected in device code
//     ("address of label extension is not supported in __device__ functions"),
//   - ptxas lowers even 128-case dense switches to BRA comparison trees
//     (see tests/jmp_big.cu), never emitting BRX/BRXU,
//   - 0 BRX/BRXU occurrences in the sampled libcublas SASS.
//
// Ground-truth encodings were therefore obtained by patching a real cubin
// instruction's opcode/operand bits and reading nvdisasm's rendering; the
// randomized validation battery lives in the docstring of tools/decode_brx.py
// (300/300 patched encodings decoded byte-exact).
//
// This kernel just provides a compilable dense switch as a placeholder / a
// reproduction of "ptxas prefers BRA trees".
__global__ void sw(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i], r = v;
    switch (v & 7) {
        case 0: r = v + 1;  break; case 1: r = v * 2; break;
        case 2: r = v - 3;  break; case 3: r = v ^ 4; break;
        case 4: r = v << 1; break; case 5: r = v >> 2; break;
        case 6: r = v * v;  break; default: r = ~v;    break;
    }
    out[i] = r;
}
