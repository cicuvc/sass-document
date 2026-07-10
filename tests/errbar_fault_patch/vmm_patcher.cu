// VMM read-only fault test. Map a region READ-only via the Virtual Memory
// Management API, pass it as the store target. A store to a *mapped* but
// non-writable page is a protection fault (translation succeeds) — possibly a
// different (later/async) fault class than an unmapped VA.
// Patch the following good-store to an illegal instruction; if the RO-store
// fault is synchronous we see 700 (illegal never runs), if async we see 715.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda.h>
#define CHECK(e) do{CUresult r=e; if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("CUDA err %s at %d\n",s,__LINE__);exit(1);} }while(0)

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s victim5.cubin [patch] [access=read|none|rw]\n",argv[0]);return 1;}
    bool patch = argc>2 && !strcmp(argv[2],"patch");
    const char* acc = argc>3?argv[3]:"read";

    // read cubin
    FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); size_t sz=ftell(f); rewind(f);
    char*buf=(char*)malloc(sz); fread(buf,1,sz,f); fclose(f);
    if(patch){
        uint8_t good_lo[]={0x86,0x79,0x00,0x04,0x07,0x00,0x00,0x00}; // 0x0000000704007986
        int found=0;
        for(size_t i=0;i+7<sz;i++) if(!memcmp(buf+i,good_lo,8)){ memset(buf+i,0,16); printf("patched good-store->illegal @0x%zx\n",i); found=1; break; }
        if(!found){printf("good-store enc not found\n");return 1;}
    }

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev,0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx,NULL,0,dev));

    // --- VMM: reserve + map a region with chosen access ---
    CUmemAllocationProp prop={}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=dev;
    size_t gran=0; CHECK(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t size=gran; // one granule
    CUmemGenericAllocationHandle h; CHECK(cuMemCreate(&h,size,&prop,0));
    CUdeviceptr ptr; CHECK(cuMemAddressReserve(&ptr,size,0,0,0));
    CHECK(cuMemMap(ptr,size,0,h,0));
    CUmemAccessDesc ad={}; ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id=dev;
    ad.flags = !strcmp(acc,"none")?CU_MEM_ACCESS_FLAGS_PROT_NONE :
               !strcmp(acc,"rw")?CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ;
    CHECK(cuMemSetAccess(ptr,size,&ad,1));
    printf("mapped RO region at 0x%llx (access=%s, granule=%zu)\n",(unsigned long long)ptr,acc,gran);

    CUdeviceptr good; CHECK(cuMemAlloc(&good,4096));

    CUmodule mod; CHECK(cuModuleLoadData(&mod,buf));
    CUfunction k; CHECK(cuModuleGetFunction(&k,mod,"victim"));
    uint32_t v=7; void*args[]={&ptr,&good,&v};
    CHECK(cuLaunchKernel(k,1,1,1,1,1,1,0,0,args,0));
    CUresult s=cuCtxSynchronize();
    if(s!=CUDA_SUCCESS){const char*n;cuGetErrorString(s,&n);printf("FAULT: %s (code %d)\n",n,(int)s);}
    else printf("completed OK\n");
    return 0;
}
