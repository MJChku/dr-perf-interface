// GPU markers and standalone translated-kernel execution. No CPU perfmark ABI.
#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#ifndef GX_DEVICE_H
#define GX_DEVICE_H
struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned a=1, unsigned b=1, unsigned c=1): x(a), y(b), z(c) {}
};
#endif

extern "C" {
uint64_t drperf_gpu_region_begin(const char* name, size_t n,
                               const char* const* names, const int64_t* values);
void drperf_gpu_region_end(uint64_t token);
void* drperf_gpu_module_load(const char* path);
void drperf_gpu_module_close(void* module);
void drperf_gpu_launch(void* module, const char* kernel, dim3 grid, dim3 block,
                       void** arguments, size_t dynamic_shared_bytes);
}

namespace drperf::gpu {
class Region {
    uint64_t token_;
public:
    explicit Region(const char* name,
                    std::initializer_list<std::pair<const char*, int64_t>> pcvs = {}) {
        std::vector<const char*> names;
        std::vector<int64_t> values;
        for (auto& p: pcvs) { names.push_back(p.first); values.push_back(p.second); }
        token_ = drperf_gpu_region_begin(name, names.size(), names.data(), values.data());
    }
    ~Region() { drperf_gpu_region_end(token_); }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;
};
class Module {
    void* handle_;
public:
    explicit Module(const char* path): handle_(drperf_gpu_module_load(path)) {}
    ~Module() { drperf_gpu_module_close(handle_); }
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    void launch(const char* kernel, dim3 grid, dim3 block, void** args,
                size_t dynamic_shared_bytes=0) {
        drperf_gpu_launch(handle_, kernel, grid, block, args, dynamic_shared_bytes);
    }
};
}
