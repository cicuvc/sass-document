#include <cstdint>
extern "C" __global__ void fence_plain(uint32_t* o){ __threadfence(); *o=1; }  // no cluster
// also: cluster launch with varying cluster dims (if supported)
