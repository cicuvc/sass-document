// AUTO-GENERATED n128 fp8 wgmma hazard probe (large accumulator + async overlap).
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define DATASETS 512
#define SLICE 8192
#define NREG 64
__device__ __forceinline__ uint64_t make_desc(uint32_t s){uint64_t d=0;d|=((uint64_t)(s&0x3FFFF)>>4);d|=((uint64_t)1)<<16;d|=((uint64_t)1)<<32;return d;}
__device__ __forceinline__ __nv_fp8_e4m3 gen(unsigned ds,int i){unsigned x=ds*2654435761u+(unsigned)i*40503u;x^=x>>15;x*=2246822519u;x^=x>>13;x*=3266489917u;x^=x>>16;return __nv_fp8_e4m3(((int)(x&0xff)-128)*(1.0f/64.0f));}
__device__ __forceinline__ void compute(unsigned ds,int M,float d[NREG]){
  extern __shared__ char sm[]; int wg=threadIdx.x>>7;
  __nv_fp8_e4m3* base=(__nv_fp8_e4m3*)(sm+wg*SLICE); int lane=threadIdx.x&127;
  for(int i=lane;i<SLICE;i+=128) base[i]=gen(ds,i);
  __syncthreads();
  uint64_t dA=make_desc(__cvta_generic_to_shared(base));
  uint64_t dB=make_desc(__cvta_generic_to_shared(base+2048));
  asm volatile("wgmma.fence.sync.aligned;\n");
  { asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 0, 1, 1;\n"
  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(dA),"l"(dB)); }                       // first: overwrite (D=A*B), no zero-init -> real async
  for(int k=1;k<M;k++){ asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1;\n"
  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(dA),"l"(dB)); }   // accumulate
  asm volatile("wgmma.commit_group.sync.aligned;\n");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}
extern "C" __global__ void ref_kernel(int M,float* ref){
  unsigned ds=blockIdx.x; float d[NREG]; compute(ds,M,d); int lane=threadIdx.x&127;
  float* r=ref+(size_t)ds*128*NREG+lane*NREG;
  for(int i=0;i<NREG;i++) r[i]=d[i];
}
extern "C" __global__ void test_kernel(int M,int Nwg,const float* ref,unsigned long long* mm,float* mx){
  int wg=threadIdx.x>>7; unsigned ds=((unsigned)blockIdx.x*Nwg+wg)%DATASETS;
  float d[NREG]; compute(ds,M,d); int lane=threadIdx.x&127;
  const float* r=ref+(size_t)ds*128*NREG+lane*NREG;
  for(int i=0;i<NREG;i++) if(__float_as_uint(d[i])!=__float_as_uint(r[i])){atomicAdd(mm,1ULL);atomicMax((int*)mx,__float_as_int(fabsf(d[i]-r[i])));}
}
int main(int argc,char** argv){
  int M=argc>1?atoi(argv[1]):32; int reps=argc>2?atoi(argv[2]):20; int maxwg=argc>3?atoi(argv[3]):6;
  int smem=maxwg*SLICE;
  cudaFuncSetAttribute(test_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  cudaFuncSetAttribute(ref_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SLICE);
  int nsm; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
  float* ref; cudaMalloc(&ref,(size_t)DATASETS*128*NREG*4);
  unsigned long long* d_mm; float* d_max; cudaMalloc(&d_mm,8); cudaMalloc(&d_max,4);
  ref_kernel<<<DATASETS,128,SLICE>>>(M,ref);
  printf("H800 fp8 n128 hazard  M=%d reps=%d SMs=%d NREG=%d ref=%s\n",M,reps,nsm,NREG,cudaGetErrorString(cudaDeviceSynchronize()));
  for(int Nwg=1;Nwg<=maxwg;Nwg++){
    int block=Nwg*128; if(block>1024) break; int grid=nsm*2;
    unsigned long long tot=0; float gmax=0;
    for(int r=0;r<reps;r++){
      cudaMemset(d_mm,0,8); cudaMemset(d_max,0,4);
      test_kernel<<<grid,block,Nwg*SLICE>>>(M,Nwg,ref,d_mm,d_max);
      if(cudaDeviceSynchronize()!=cudaSuccess){printf("  Nwg=%d ERR %s\n",Nwg,cudaGetErrorString(cudaGetLastError()));break;}
      unsigned long long m; float x; cudaMemcpy(&m,d_mm,8,cudaMemcpyDeviceToHost); cudaMemcpy(&x,d_max,4,cudaMemcpyDeviceToHost);
      tot+=m; if(x>gmax)gmax=x;
    }
    double checked=(double)reps*grid*block*NREG;
    printf("  wgs=%d  mismatches=%llu / %.0f (%.2e)  maxdev=%g%s\n",Nwg,tot,checked,tot/checked,gmax,tot?"   <-- HAZARD":"");
  }
  return 0;
}
