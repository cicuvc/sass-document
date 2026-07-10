
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda.h>
#define CHECK(e) do{CUresult r=e;if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("CUDA err %s at %d\n",s,__LINE__);exit(1);}}while(0)

int main(int argc,char**argv){
    if(argc<3){printf("usage: %s cubin tgt\n",argv[0]);return 1;}
    const char* tgt=argv[2];
    struct{const char*n;uint64_t lo;}tbl[]={
     {"membar",0x7992ULL},{"errbar",0x79abULL},{"cgaerrbar",0x75abULL},
     {"cctl",0xff00798fULL}};
    uint64_t enc=0; for(auto&t:tbl) if(!strcmp(tgt,t.n)) enc=t.lo;

    FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); size_t sz=ftell(f); rewind(f);
    char*buf=(char*)malloc(sz); fread(buf,1,sz,f); fclose(f);
    if(enc){uint8_t k[8]; for(int i=0;i<8;i++)k[i]=(enc>>(8*i))&0xff;
        int found=0; for(size_t i=0;i+7<sz;i++) if(!memcmp(buf+i,k,8)){memset(buf+i,0,16);found=1;printf("patched %s@0x%zx\n",tgt,i);break;}
        if(!found){printf("enc for %s not found\n",tgt);return 1;}}else printf("no patch\n");

    CHECK(cuInit(0)); CUdevice dev; CHECK(cuDeviceGet(&dev,0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx,NULL,0,dev));
    CUmodule mod; CHECK(cuModuleLoadData(&mod,buf));
    CUfunction k; CHECK(cuModuleGetFunction(&k,mod,"victim"));
    uint32_t* hbuf; cudaHostAlloc(&hbuf,64,cudaHostAllocMapped); for(int i=0;i<16;i++)hbuf[i]=0;
    volatile uint32_t* dh; cudaHostGetDevicePointer((void**)&dh,hbuf,0);
    uint32_t* good; cudaMalloc(&good,4096);
    int bad_rank=7;
    void*args[]={(void*)&dh,(void*)&good,(void*)&bad_rank};
    CUlaunchAttribute attrs[1];
    attrs[0].id=CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[0].value.clusterDim={2,1,1};
    CUlaunchConfig cfg={};
    cfg.gridDimX=2; cfg.gridDimY=1; cfg.gridDimZ=1;
    cfg.blockDimX=1; cfg.blockDimY=1; cfg.blockDimZ=1;
    cfg.hStream=0; cfg.attrs=attrs; cfg.numAttrs=1;
    CHECK(cuLaunchKernelEx(&cfg,k,args,NULL));
    CUresult s=cuCtxSynchronize();
    if(s!=CUDA_SUCCESS){const char*n;cuGetErrorString(s,&n);printf("fault: %s (%d)\n",n,(int)s);}else printf("ok\n");
    return 0;
}
