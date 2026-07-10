// SASS binary patcher: replace ERRBAR or CGAERRBAR in the victim cubin with
// an illegal instruction (all zeros), then launch via Driver API. Observe the
// reported error to determine whether ERRBAR is the fault-collection point.
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cstring>
#include <cstdlib>

// CUDA Driver API error helpers
#define CHECK(e) do { CUresult r=e; if(r!=CUDA_SUCCESS){ const char*s; cuGetErrorString(r,&s); printf("CUDA error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1); } } while(0)

int main(int argc, char**argv){
    if(argc<2){ printf("usage: %s <victim.cubin> [patch_pos]\n  patch_pos: none|errbar|cgaerrbar\n",argv[0]); return 1; }
    const char* patch_pos = argc>2 ? argv[2] : "none";

    FILE* f=fopen(argv[1],"rb"); if(!f){ perror(argv[1]); return 1; }
    fseek(f,0,SEEK_END); size_t sz=ftell(f); rewind(f);
    char* buf=(char*)malloc(sz); fread(buf,1,sz,f); fclose(f);

    // Search for ERRBAR encoding (lo64 only — 8 bytes)
    uint8_t errbar_lo[] = {0xab,0x79,0x00,0x00,0x00,0x00,0x00,0x00}; // little-endian 0x00000000000079ab
    uint8_t cgaerrbar_lo[] = {0xab,0x75,0x00,0x00,0x00,0x00,0x00,0x00}; // 0x00000000000075ab
    uint8_t membar_lo[]   = {0x92,0x79,0x00,0x00,0x00,0x00,0x00,0x00}; // 0x0000000000007992
    uint8_t cctl_lo[]     = {0x8f,0x79,0x00,0xff,0x00,0x00,0x00,0x00}; // 0x00000000ff00798f
    uint8_t zero16[16] = {0};

    char* target = nullptr;
    if(!strcmp(patch_pos,"errbar"))   target = (char*)errbar_lo;
    else if(!strcmp(patch_pos,"cgaerrbar")) target = (char*)cgaerrbar_lo;
    else if(!strcmp(patch_pos,"membar")) target = (char*)membar_lo;
    else if(!strcmp(patch_pos,"cctl")) target = (char*)cctl_lo;
    else if(!strcmp(patch_pos,"hex")){
        static uint8_t hx[8]; unsigned long long v=strtoull(argv[3],0,16);
        for(int k=0;k<8;k++) hx[k]=(v>>(8*k))&0xff;
        target=(char*)hx;
    }
    int found=0;

    if(target){
        // Search for ERRBAR or CGAERRBAR by their lo64, overwrite the full 16B
        for(size_t i=0; i+7<sz; i++){
            if(!memcmp(buf+i, target, 8)){
                // Found the lo64 at offset i; the full instruction is 16B starting at i
                size_t insn_off = i;
                printf("patched %s at file offset 0x%zx\n", patch_pos, insn_off);
                memcpy(buf+insn_off, zero16, 16);
                found=1; break;
            }
        }
        if(!found){ printf("ERROR: %s encoding not found in cubin\n", patch_pos); return 1; }
    } else { printf("no patch applied\n"); }

    // --- Driver API: load patched cubin and launch ---
    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev,0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx,NULL,0,dev));

    CUmodule mod; CHECK(cuModuleLoadData(&mod,buf));

    CUfunction kernel; CHECK(cuModuleGetFunction(&kernel,mod,"victim"));

    // Allocate a valid global buffer for the second store
    CUdeviceptr d_good; CHECK(cuMemAlloc(&d_good,4096));
    uint32_t good_h[4]={0};
    uint32_t val=42;

    void* args[] = {&d_good, &val};
    CHECK(cuLaunchKernel(kernel, 1,1,1, 1,1,1, 0,0, args,0));

    // Check for launch error
    CUresult sync = cuCtxSynchronize();
    if(sync!=CUDA_SUCCESS){
        const char* s; cuGetErrorString(sync,&s);
        printf("kernel fault: %s\n", s);
        // 700 = CUDA_ERROR_ILLEGAL_INSTRUCTION
        // 700? or 719? Let's print the numeric code
        printf("  error code = %d\n", (int)sync);
    } else {
        printf("kernel completed successfully (unexpected)\n");
        CUresult e=cuMemcpyDtoH(good_h,d_good,16);
        printf("good[0]=%u\n",good_h[0]);
    }

    CHECK(cuCtxDestroy(ctx));
    free(buf);
    return 0;
}
