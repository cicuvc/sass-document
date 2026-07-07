__global__ void base(int *a,int *o){int i=threadIdx.x;o[i]=a[i]+1;}
__global__ void ld(int *a,int *o){int i=threadIdx.x;asm volatile("griddepcontrol.launch_dependents;");o[i]=a[i]+1;}
__global__ void wt(int *a,int *o){int i=threadIdx.x;asm volatile("griddepcontrol.wait;");o[i]=a[i]+1;}
