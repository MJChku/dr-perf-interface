// Optional native CUDA device-link implementation. Keep this separate from
// the annotation header: whole-program optimization would specialize away the
// marker arguments, destroying the PTX instrumentation ABI.
#include "drperf_gpu_device.cuh"
extern "C" __device__ __noinline__ void __drperf_gpu_device_begin_v2(const char*,unsigned) {}
extern "C" __device__ __noinline__ void __drperf_gpu_device_pcv_v1(const char*,int64_t) {}
extern "C" __device__ __noinline__ void __drperf_gpu_device_end_v1() {}

extern "C" __device__ __noinline__ void __drperf_gpu_device_ready_v2() {}
