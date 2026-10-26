// GPU markers and standalone translated-kernel execution. No CPU perfmark ABI.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#if defined(__CUDACC__) || defined(__VECTOR_TYPES_H__)
// Drivers compiled by nvcc (e.g. to reuse a library's host-side parameter
// structs) use CUDA's dim3: the same three-unsigned layout the runtime expects.
#include <vector_types.h>
#elif !defined(GX_DEVICE_H)
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
void drperf_gpu_launch_cluster(void* module, const char* kernel, dim3 grid, dim3 block,
                       void** arguments, size_t dynamic_shared_bytes, dim3 cluster);
// Read-before-write checking (initcheck): device reads of bytes in a range
// marked uninitialized that no device write or host mark has initialized are
// counted as uninitialized_read_bytes at the reading site.
void drperf_gpu_mark_uninitialized(const void* p, size_t bytes, const char* name);
void drperf_gpu_mark_initialized(const void* p, size_t bytes);
void drperf_gpu_forget_buffer(const void* p);
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
    void launch_cluster(const char* kernel, dim3 grid, dim3 block, dim3 cluster,
                        void** args, size_t dynamic_shared_bytes=0) {
        drperf_gpu_launch_cluster(handle_,kernel,grid,block,args,dynamic_shared_bytes,cluster);
    }
};
// Device-like buffer: 256-byte aligned like cudaMalloc. Sector/line footprint
// metrics depend on allocation alignment; std::vector storage is only 16-byte
// aligned, so footprints of the same kernel shift with the host allocator.
template <typename T> class Buffer {
    T* data_ = nullptr;
    size_t size_ = 0;
    bool tracked_ = false;
public:
    Buffer() = default;
    explicit Buffer(size_t n, T fill = T()) : size_(n) {
        size_t bytes = ((n ? n : 1) * sizeof(T) + 255) / 256 * 256;
        data_ = static_cast<T*>(std::aligned_alloc(256, bytes));
        if (!data_) throw std::bad_alloc();
        for (size_t i = 0; i < n; ++i) data_[i] = fill;
    }
    // Contents deliberately left undefined, like fresh cudaMalloc memory;
    // device reads before any write are reported (uninitialized_read_bytes).
    static Buffer uninitialized(size_t n, const char* name = nullptr) {
        Buffer b; b.size_ = n;
        size_t bytes = ((n ? n : 1) * sizeof(T) + 255) / 256 * 256;
        b.data_ = static_cast<T*>(std::aligned_alloc(256, bytes));
        if (!b.data_) throw std::bad_alloc();
        std::memset(b.data_, 0xA5, bytes);
        drperf_gpu_mark_uninitialized(b.data_, n * sizeof(T), name);
        b.tracked_ = true;
        return b;
    }
    // Host writes after construction (e.g. filling inputs) must be declared.
    void host_written(size_t first = 0, size_t count = size_t(-1)) {
        if (!tracked_) return;
        if (count > size_ - first) count = size_ - first;
        drperf_gpu_mark_initialized(data_ + first, count * sizeof(T));
    }
    ~Buffer() { if (tracked_) drperf_gpu_forget_buffer(data_); std::free(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : data_(o.data_), size_(o.size_), tracked_(o.tracked_) { o.data_ = nullptr; o.size_ = 0; o.tracked_ = false; }
    Buffer& operator=(Buffer&& o) noexcept { std::swap(data_, o.data_); std::swap(size_, o.size_); std::swap(tracked_, o.tracked_); return *this; }
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    T* begin() { return data_; } T* end() { return data_ + size_; }
    T& front() { return data_[0]; } T& back() { return data_[size_ - 1]; }
};
}
