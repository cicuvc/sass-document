__global__ void __cluster_dims__(2,1,1) k(int *a, int *out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] += 1;
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    out[i] = a[i];
}
// relaxed / aligned variants
__global__ void __cluster_dims__(2,1,1) k2(int *a, int *out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] += 1;
    asm volatile("barrier.cluster.arrive.relaxed;");
    asm volatile("barrier.cluster.wait.acquire;");
    out[i] = a[i];
}
