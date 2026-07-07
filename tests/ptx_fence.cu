__global__ void __cluster_dims__(2,1,1) k(int *a, int *out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] += 1;
    asm volatile("fence.cluster.acq_rel;");
    out[i] = a[i];
}
__global__ void kcmp(int *a, int *out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] += 1;
    asm volatile("fence.acq_rel.gpu;");        // for comparison
    asm volatile("fence.sc.cluster;");
    out[i] = a[i];
}
