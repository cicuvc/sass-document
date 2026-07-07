__global__ void k(int *a, int *out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] += 1;
    asm volatile("griddepcontrol.launch_dependents;");
    out[i] = a[i];
}
__global__ void kw(int *a, int *out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    asm volatile("griddepcontrol.wait;");
    out[i] = a[i] + 1;
}
