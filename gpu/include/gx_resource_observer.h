// Optional GX resource observer ABI. No CUDA interposition or emulator types.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define GX_RESOURCE_OBSERVER_ABI 1
struct gx_resource_observer_v1 {
    uint32_t abi;
    // Called on the submitting application thread, after graph-capture routing.
    void* (*submit)(const char* kernel, int translated);
    // Executed PTX resource counts; site includes translator basic-block ID.
    void (*count)(void* launch, const char* site, const char* metric, uint64_t amount);
    // Exactly once per submitted launch; completed=0 includes cancellation.
    void (*finish)(void* launch, int completed);
};
const struct gx_resource_observer_v1* drperf_gpu_gx_observer_v1(void);
#ifdef __cplusplus
}
#endif
