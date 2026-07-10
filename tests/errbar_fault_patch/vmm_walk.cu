// VMM RO-store + configurable illegal-instruction position, to locate where the
// ASYNC permission fault is collected. Walk the illegal instr through the fence
// block; the 715->700 transition marks the collection instruction.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda.h>
#define CHECK(e) do{CUresult r=e; if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("CUDA err %s at %d\n",s,__LINE__);exit(1);} }while(0)

struct T{const char*name; uint64_t lo;};
int main(int argc,char**argv){
    if(argc<3){printf("usage: %s cubin none|membar|errbar|cgaerrbar|cctl|goodstore [access=read]\n",argv[0]);return 1;}
    const char* tgt=argv[2];
    const char* acc=argc>3?argv[3]:"read";
    T tbl[]={{"membar",0x7992ULL},{"errbar",0x79abULL},{"cgaerrbar",0x75abULL},
             {"cctl",0xff00798fULL},{"goodstore",0x0000000704007986ULL}};
    uint64_t enc=0; for(auto&t:tbl) if(!strcmp(tgt,t.name)) enc=t.lo;

    FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); size_t sz=ftell(f); rewind(f);
    char*buf=(char*)malloc(sz); fread(buf,1,sz,f); fclose(f);
    if(enc){ uint8_t k[8]; for(int i=0;i<8;i++)k[i]=(enc>>(8*i))&0xff;
        int found=0; for(size_t i=0;i+7<sz;i++) if(!memcmp(buf+i,k,8)){memset(buf+i,0,16);found=1;printf("patched %s@0x%zx  ",tgt,i);break;}
        if(!found){printf("enc for %s not found\n",tgt);return 1;} }
    else printf("no patch  ");
    if(argc>4 && !strcmp(argv[4],"nopmembar")){
        uint8_t mb[8]={0x92,0x79,0,0,0,0,0,0}; uint8_t nop[8]={0x18,0x79,0,0,0,0,0,0};
        for(size_t i=0;i+7<sz;i++) if(!memcmp(buf+i,mb,8)){ memcpy(buf+i,nop,8); printf("(membar->NOP@0x%zx) ",i); break; }
    }

    CHECK(cuInit(0)); CUdevice dev; CHECK(cuDeviceGet(&dev,0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx,NULL,0,dev));
    CUmemAllocationProp prop={}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=dev;
    size_t gran; CHECK(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CUmemGenericAllocationHandle h; CHECK(cuMemCreate(&h,gran,&prop,0));
    CUdeviceptr ptr; CHECK(cuMemAddressReserve(&ptr,gran,0,0,0));
    CHECK(cuMemMap(ptr,gran,0,h,0));
    CUmemAccessDesc ad={}; ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id=dev;
    ad.flags = !strcmp(acc,"none")?CU_MEM_ACCESS_FLAGS_PROT_NONE:!strcmp(acc,"rw")?CU_MEM_ACCESS_FLAGS_PROT_READWRITE:CU_MEM_ACCESS_FLAGS_PROT_READ;
    CHECK(cuMemSetAccess(ptr,gran,&ad,1));
    CUdeviceptr good; CHECK(cuMemAlloc(&good,4096));
    CUmodule mod; CHECK(cuModuleLoadData(&mod,buf));
    CUfunction k; CHECK(cuModuleGetFunction(&k,mod,"victim"));
    uint32_t v=7; void*args[]={&ptr,&good,&v};
    CHECK(cuLaunchKernel(k,1,1,1,1,1,1,0,0,args,0));
    CUresult s=cuCtxSynchronize();
    if(s!=CUDA_SUCCESS){const char*n;cuGetErrorString(s,&n);printf("=> %s (%d)\n",n,(int)s);}
    else printf("=> completed OK\n");
    return 0;
}
