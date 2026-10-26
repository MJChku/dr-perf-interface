#pragma once
// Device markers for the standalone instrumented PTX runner. For native CUDA
// builds define DRPERF_GPU_DISABLE_DEVICE_REGIONS, or link device_marker_stubs.cu
// with relocatable device code. Declarations preserve the marker ABI in NVCC PTX.
#include <stdint.h>
#include <type_traits>
#include <initializer_list>
#ifndef __CUDACC__
#error "drperf_gpu_device.cuh requires CUDA; use drperf_gpu.h for host launch regions"
#endif
extern "C" {
__device__ __noinline__ void __drperf_gpu_device_begin_v2(const char*, unsigned);
__device__ __noinline__ void __drperf_gpu_device_pcv_v1(const char*, int64_t);
__device__ __noinline__ void __drperf_gpu_device_end_v1();
__device__ __noinline__ void __drperf_gpu_device_ready_v2();
}
namespace drperf { namespace gpu {
// Cooperative scopes aggregate all physical lanes of a warp/block. All members
// must enter the same named occurrence with identical PCVs and nesting.
enum class DeviceScope : unsigned { lane=0, warp=1, block=2 };
__device__ __forceinline__ void device_begin(const char* name, DeviceScope scope) {
#ifndef DRPERF_GPU_DISABLE_DEVICE_REGIONS
    __drperf_gpu_device_begin_v2(name, unsigned(scope));
#endif
}
template<class T>
__device__ __forceinline__ void device_pcv(const char* name, T value) {
    static_assert(std::is_integral<T>::value && sizeof(T)<=8,
                  "GPU PCVs must be integer expressions of at most 64 bits");
#ifndef DRPERF_GPU_DISABLE_DEVICE_REGIONS
    if (std::is_unsigned<T>::value && uint64_t(value)>0x7fffffffffffffffULL)
        asm volatile("trap;");
    __drperf_gpu_device_pcv_v1(name, int64_t(value));
#endif
}
__device__ __forceinline__ void device_end() {
#ifndef DRPERF_GPU_DISABLE_DEVICE_REGIONS
    __drperf_gpu_device_end_v1();
#endif
}
__device__ __forceinline__ void device_ready() {
#ifndef DRPERF_GPU_DISABLE_DEVICE_REGIONS
    __drperf_gpu_device_ready_v2();
#endif
}
// All PCV expressions are evaluated before entry. No incremental .pcv()
// mutator is exposed: a result computed inside the body cannot be retrofitted
// as an entry PCV. The low-level registration calls are constructor machinery.
struct DevicePCV {
    const char* name;
    int64_t value;
    template<class T> __device__ __forceinline__ DevicePCV(const char* n,T v):name(n),value(v) {
        static_assert(std::is_integral<T>::value && sizeof(T)<=8,
                      "GPU PCVs must be integer expressions of at most 64 bits");
#ifndef DRPERF_GPU_DISABLE_DEVICE_REGIONS
        if (std::is_unsigned<T>::value && uint64_t(v)>0x7fffffffffffffffULL) asm volatile("trap;");
#endif
    }
};
__device__ __forceinline__ void device_enter(const char* name,DeviceScope scope,
                                            std::initializer_list<DevicePCV> pcvs={}) {
#ifndef DRPERF_GPU_DISABLE_DEVICE_REGIONS
    device_begin(name,scope);
    for (const auto& p:pcvs) device_pcv(p.name,p.value);
    device_ready();
#endif
}
class DeviceRegion {
public:
    __device__ __forceinline__ explicit DeviceRegion(
        const char* name, DeviceScope scope=DeviceScope::lane,
        std::initializer_list<DevicePCV> pcvs={}) { device_enter(name,scope,pcvs); }
    __device__ __forceinline__ DeviceRegion(
        const char* name,std::initializer_list<DevicePCV> pcvs) { device_enter(name,DeviceScope::lane,pcvs); }
    __device__ __forceinline__ ~DeviceRegion() { device_end(); }
    DeviceRegion(const DeviceRegion&)=delete;
    DeviceRegion& operator=(const DeviceRegion&)=delete;
};
}}
#define DRPERF_GPU_DEVICE_BEGIN(name,scope,...) ::drperf::gpu::device_enter((name),(scope),{__VA_ARGS__})
#define DRPERF_GPU_DEVICE_END() ::drperf::gpu::device_end()
