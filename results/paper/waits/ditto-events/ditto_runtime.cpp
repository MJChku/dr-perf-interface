#include "perfmark.h"
#include <atomic>
#include "ditto_runtime.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../cpu/lz77/lz77.h"

namespace {
std::atomic<uint64_t> annotation_next_event{1};

constexpr uint32_t kAbiVersion = 2;
constexpr size_t kCandidateCount = 7;

struct Profile {
    const char* name;
    uint32_t graph_id;
    uint16_t version;
    const char* quantizer;
    const char* codec;
    uint32_t transforms;
    uint32_t requirements;
    uint8_t symmetric;
    const char* description;
    uint8_t per_head = 0;
};

constexpr uint32_t S = DITTO_TRANSFORM_FP8_SPLIT;
constexpr uint32_t D = DITTO_TRANSFORM_DEROPE;
constexpr uint32_t V = DITTO_TRANSFORM_VHALF;
constexpr uint32_t P = DITTO_REQUIRES_POSITIONS;
constexpr uint32_t W = DITTO_REQUIRES_MODEL_WEIGHTS;

/* This table is the deployed compressor registry's single source of truth. */
constexpr Profile kProfiles[] = {
    {"fp8-split-ditto", 0x1001, 1, "identity_fp8", "ditto_rans", S, 0, 0,
     "E4M3 exponent Ditto; sign and mantissa literal"},
    {"fp8-derope-split-ditto", 0x1002, 1, "identity_fp8", "ditto_rans",
     S | D, P, 0, "inverse RoPE K; E4M3 exponent Ditto"},
    {"fp8-split-raw", 0x1003, 1, "identity_fp8", "raw", S, 0, 0, ""},
    {"fp8-derope-split-raw", 0x1004, 1, "identity_fp8", "raw", S | D, P,
     0, ""},
    {"fp8-split-order0", 0x1005, 1, "identity_fp8", "order0_ans", S, 0, 0,
     "E4M3 exponent order-0 ANS; sign and mantissa literal"},
    {"fp8-derope-split-order0", 0x1007, 1, "identity_fp8", "order0_ans",
     S | D, P, 0, "inverse RoPE K; E4M3 exponent order-0 ANS"},
    {"fp8-raw", 0x1009, 1, "identity_fp8", "raw", 0, 0, 0,
     "literal E4M3 bytes; no transform or compression"},

    {"int4-order0", 0x2001, 1, "grouped_int4", "order0_ans", 0, 0, 0, ""},
    {"int4-raw", 0x2002, 1, "grouped_int4", "raw", 0, 0, 0, ""},
    {"int4-ditto-fullrow", 0x2003, 1, "grouped_int4", "ditto_rans", 0, 0,
     0, ""},
    {"int4-vhalf-ditto", 0x2004, 1, "grouped_int4", "ditto_rans", V,
     P | W, 1, ""},
    {"int4-vhalf-raw", 0x2006, 1, "grouped_int4", "raw", V, P | W, 1, ""},
    {"int4-derope-ditto", 0x2008, 1, "grouped_int4", "ditto_rans", D, P,
     0, ""},
    {"int4-derope-order0", 0x2009, 1, "grouped_int4", "order0_ans", D, P,
     0, ""},
    {"int4-derope-raw", 0x200B, 1, "grouped_int4", "raw", D, P, 0, ""},
    {"int4-sym-order0", 0x200C, 1, "grouped_int4", "order0_ans", 0, 0, 1,
     ""},
    {"int4-sym-ditto", 0x200D, 1, "grouped_int4", "ditto_rans", 0, 0, 1,
     ""},
    {"int4-sym-raw", 0x200E, 1, "grouped_int4", "raw", 0, 0, 1, ""},
    {"int4-vhalf-order0", 0x200F, 1, "grouped_int4", "order0_ans", V,
     P | W, 1, ""},
    {"int4-derope-sym-ditto", 0x2010, 1, "grouped_int4", "ditto_rans", D,
     P, 1, ""},
    {"int4-derope-sym-order0", 0x2011, 1, "grouped_int4", "order0_ans", D,
     P, 1, ""},
    {"int4-derope-sym-raw", 0x2013, 1, "grouped_int4", "raw", D, P, 1, ""},
    {"int4-head-raw", 0x2014, 1, "grouped_int4", "raw", 0, 0, 0,
     "per-token, per-head INT4; literal codes", 1},
    {"int4-head-ditto", 0x2015, 1, "grouped_int4", "ditto_rans", 0, 0, 0,
     "per-token, per-head INT4; Ditto-coded codes", 1},
    /* No derope-raw twin: de-rope only helps a downstream entropy coder;
       literal codes are the same size either way, so the raw control stays
       int4-head-raw. */
    {"int4-head-derope-ditto", 0x2016, 1, "grouped_int4", "ditto_rans", D, P,
     0, "per-token, per-head INT4 on de-roped K; Ditto-coded codes", 1},
    {"int4-head-sym-raw", 0x2017, 1, "grouped_int4", "raw", 0, 0, 1,
     "per-token, per-head symmetric INT4; literal codes", 1},
    {"int4-head-sym-ditto", 0x2018, 1, "grouped_int4", "ditto_rans", 0, 0, 1,
     "per-token, per-head symmetric INT4; Ditto-coded codes", 1},
    {"int4-head-sym-derope-ditto", 0x2019, 1, "grouped_int4", "ditto_rans",
     D, P, 1,
     "per-token, per-head symmetric INT4 on de-roped K; Ditto-coded codes",
     1},
    {"int4-tok-ditto", 0x201A, 1, "grouped_int4", "ditto_rans", 0, 0, 0,
     "per-token INT4: one D-wide scale group shared by all KV heads; "
     "Ditto-coded codes"},
    {"int4-tok-sym-ditto", 0x201B, 1, "grouped_int4", "ditto_rans", 0, 0, 1,
     "per-token symmetric INT4: one D-wide scale group shared by all KV "
     "heads; Ditto-coded codes"},
    {"int4-tok-derope-ditto", 0x201C, 1, "grouped_int4", "ditto_rans", D, P,
     0, "per-token INT4 on de-roped K: one D-wide scale group shared by all "
     "KV heads; Ditto-coded codes"},
    {"int4-head-vhalf-ditto", 0x201D, 1, "grouped_int4", "ditto_rans", V,
     P | W, 1, "per-token, per-head symmetric INT4 on V; K reconstructed; "
     "Ditto-coded codes", 1},
    /* Sliding-window groups: K != V there, so the full K+V row is stored and
       no de-rope/K-reconstruction applies. Order-0 keeps the per-head codes
       cheap enough to run on 40 of gemma-4's 48 layers. */
    {"int4-head-order0", 0x201E, 1, "grouped_int4", "order0_ans", 0, 0, 0,
     "per-token, per-head INT4; order-0 ANS codes", 1},
};

void set_report(ditto_report* report, int code, const char* message) {
    if (!report) return;
    report->code = code;
    std::snprintf(report->message, sizeof(report->message), "%s",
                  message ? message : "");
}

int fail(ditto_report* report, int code, const std::string& message) {
    set_report(report, code, message.c_str());
    return code;
}

void copy_profile(const Profile& p, ditto_profile_info* out) {
    out->name = p.name;
    out->graph_id = p.graph_id;
    out->version = p.version;
    out->quantizer = p.quantizer;
    out->codec = p.codec;
    out->transforms = p.transforms;
    out->requirements = p.requirements;
    out->symmetric = p.symmetric;
    out->description = p.description;
    out->per_head = p.per_head;
}

void validate_registry() {
    for (size_t i = 0; i < std::size(kProfiles); ++i) {
        const Profile& p = kProfiles[i];
        if (!p.name || !*p.name || !p.graph_id || !p.version)
            throw std::logic_error("invalid compressor identity");
        const bool fp8 = std::strcmp(p.quantizer, "identity_fp8") == 0;
        const bool split = (p.transforms & S) != 0;
        const bool derope = (p.transforms & D) != 0;
        const bool vhalf = (p.transforms & V) != 0;
        if (p.per_head > 1 || (p.per_head && fp8))
            throw std::logic_error(std::string(p.name) +
                                   ": per-head grouping requires grouped_int4");
        if (!fp8 && split)
            throw std::logic_error(std::string(p.name) +
                                   ": fp8_split requires identity_fp8");
        if (fp8 && !split && std::strcmp(p.codec, "raw") != 0)
            throw std::logic_error(std::string(p.name) +
                                   ": unsplit identity_fp8 requires raw");
        if (derope && vhalf)
            throw std::logic_error(std::string(p.name) +
                                   ": derope and vhalf are exclusive");
        if (vhalf && (!p.symmetric ||
                      (p.requirements & (P | W)) != (P | W)))
            throw std::logic_error(std::string(p.name) +
                                   ": vhalf requires symmetric + positions + weights");
        for (size_t j = 0; j < i; ++j) {
            const Profile& q = kProfiles[j];
            if ((p.graph_id == q.graph_id && p.version == q.version) ||
                (std::strcmp(p.name, q.name) == 0 && p.version == q.version))
                throw std::logic_error("duplicate compressor identity");
        }
    }
}

void registry_once() {
    static const bool checked = [] { validate_registry(); return true; }();
    (void)checked;
}

std::vector<int32_t> build_candidates(const std::vector<int32_t>& tokens,
                                      int min_match) {
    const size_t T = tokens.size();
    std::vector<int64_t> wide(T);
    for (size_t i = 0; i < T; ++i) wide[i] = tokens[i];
    std::vector<int32_t> refs(T, -1);
    const auto matches = lz77(wide.data(), T, 3, min_match);
    for (const auto& match : matches)
        for (uint32_t i = 0; i < match.length; ++i)
            refs[match.start + i] = static_cast<int32_t>(match.ref + i);
    std::unordered_map<int64_t, std::array<int32_t, 4>> ring;
    std::vector<int32_t> out(T * kCandidateCount, -1);
    for (size_t t = 0; t < T; ++t) {
        const auto it = ring.find(wide[t]);
        const std::array<int32_t, 4> empty{-1, -1, -1, -1};
        const auto occ = it == ring.end() ? empty : it->second;
        out[t * kCandidateCount] = refs[t];
        for (size_t i = 0; i < 4; ++i)
            out[t * kCandidateCount + i + 1] = occ[i];
        out[t * kCandidateCount + 5] = t ? static_cast<int32_t>(t - 1) : -1;
        ring[wide[t]] = {static_cast<int32_t>(t), occ[0], occ[1], occ[2]};
    }
    return out;
}

class ThreadPool {
public:
    explicit ThreadPool(size_t count) {
        count = std::max<size_t>(1, count);
        for (size_t i = 0; i < count; ++i) {
            workers_.emplace_back([this] {
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mu_);
                        cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
                        if (stop_ && jobs_.empty()) return;
                        job = std::move(jobs_.front());
                        jobs_.pop();
                    }
                    job();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    template <class F>
    auto submit(F&& fn) -> std::shared_future<decltype(fn())> {
        using Result = decltype(fn());
        auto task = std::make_shared<std::packaged_task<Result()>>(
            std::forward<F>(fn));
        auto future = task->get_future().share();
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stop_) throw std::runtime_error("native scheduler stopped");
            jobs_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return future;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> jobs_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

std::string candidate_key(const int32_t* tokens, size_t count,
                          int32_t min_match) {
    std::string key(reinterpret_cast<const char*>(tokens),
                    count * sizeof(int32_t));
    key.append(reinterpret_cast<const char*>(&min_match), sizeof(min_match));
    return key;
}

using CUresult = int;
using CUfunction = void*;
using CUstream = void*;
constexpr CUresult CUDA_SUCCESS = 0;

struct CudaDriver {
    using Launch = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned, unsigned,
                                CUstream, void**, void**);
    using ErrorName = CUresult (*)(CUresult, const char**);
    using ErrorString = CUresult (*)(CUresult, const char**);

    void* library = nullptr;
    Launch launch = nullptr;
    ErrorName error_name = nullptr;
    ErrorString error_string = nullptr;

    CudaDriver() {
        library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(
            std::string("cannot load CUDA driver: ") + dlerror());
        launch = reinterpret_cast<Launch>(dlsym(library, "cuLaunchKernel"));
        error_name = reinterpret_cast<ErrorName>(
            dlsym(library, "cuGetErrorName"));
        error_string = reinterpret_cast<ErrorString>(
            dlsym(library, "cuGetErrorString"));
        if (!launch) throw std::runtime_error("CUDA driver lacks cuLaunchKernel");
    }

    ~CudaDriver() { if (library) dlclose(library); }

    std::string message(CUresult status) const {
        const char* name = nullptr;
        const char* detail = nullptr;
        if (error_name) error_name(status, &name);
        if (error_string) error_string(status, &detail);
        return std::string(name ? name : "CUDA_ERROR") +
            (detail ? std::string(": ") + detail : "");
    }
};

CudaDriver& cuda_driver() {
    static CudaDriver driver;
    return driver;
}

}  // namespace

struct ditto_context {
    using Result = std::vector<int32_t>;
    using Future = std::shared_future<Result>;

    explicit ditto_context(uint32_t workers, uint32_t limit)
        : pool(workers), cache_limit(std::max<uint32_t>(1, limit)) {}

    Future submit(const int32_t* tokens, size_t count, int32_t min_match, uint64_t* event_out = nullptr) {
        const std::string key = candidate_key(tokens, count, min_match);
        std::lock_guard<std::mutex> lock(mu);
        auto found = cache.find(key);
        if (found != cache.end()) {
            lru.splice(lru.end(), lru, found->second.second);
            if (event_out) *event_out = annotation_events.at(key);
            return found->second.first;
        }
        std::vector<int32_t> copy(tokens, tokens + count);
        const uint64_t event = annotation_next_event++;
        annotation_events[key] = event;
        if (event_out) *event_out = event;
        Future future = pool.submit(
            [tokens = std::move(copy), min_match, event]() {
                perfmark_begin("candidates.build", "tokens", tokens.size());
                auto result = build_candidates(tokens, min_match);
                perfmark_event_publish(event, 1);
                perfmark_end("candidates.build");
                return result;
            });
        lru.push_back(key);
        cache.emplace(key, std::make_pair(future, std::prev(lru.end())));
        while (cache.size() > cache_limit) {
            const std::string victim = lru.front();
            lru.pop_front();
            cache.erase(victim);
            annotation_events.erase(victim);
        }
        return future;
    }

    std::unordered_map<std::string, uint64_t> annotation_events;
    ThreadPool pool;
    size_t cache_limit;
    std::mutex mu;
    std::list<std::string> lru;
    std::unordered_map<std::string,
        std::pair<Future, std::list<std::string>::iterator>> cache;
};

extern "C" {

uint32_t ditto_runtime_abi_version(void) { return kAbiVersion; }

size_t ditto_profile_count(void) {
    registry_once();
    return std::size(kProfiles);
}

int ditto_profile_at(size_t index, ditto_profile_info* out,
                     ditto_report* report) {
    try {
        registry_once();
        if (!out) return fail(report, DITTO_INVALID_ARGUMENT, "out is null");
        if (index >= std::size(kProfiles))
            return fail(report, DITTO_NOT_FOUND, "profile index out of range");
        copy_profile(kProfiles[index], out);
        set_report(report, DITTO_OK, "");
        return DITTO_OK;
    } catch (const std::exception& e) {
        return fail(report, DITTO_INTERNAL_ERROR, e.what());
    }
}

int ditto_profile_by_name(const char* name, uint16_t version,
                          ditto_profile_info* out, ditto_report* report) {
    try {
        registry_once();
        if (!name || !out)
            return fail(report, DITTO_INVALID_ARGUMENT, "name/out is null");
        const Profile* best = nullptr;
        for (const auto& profile : kProfiles) {
            if (std::strcmp(profile.name, name) != 0) continue;
            if (version && profile.version != version) continue;
            if (!best || profile.version > best->version) best = &profile;
        }
        if (!best) return fail(report, DITTO_NOT_FOUND,
                               std::string("unknown compressor ") + name);
        copy_profile(*best, out);
        set_report(report, DITTO_OK, "");
        return DITTO_OK;
    } catch (const std::exception& e) {
        return fail(report, DITTO_INTERNAL_ERROR, e.what());
    }
}

int ditto_profile_by_ref(uint32_t graph_id, uint16_t version,
                         ditto_profile_info* out, ditto_report* report) {
    try {
        registry_once();
        if (!out) return fail(report, DITTO_INVALID_ARGUMENT, "out is null");
        for (const auto& profile : kProfiles) {
            if (profile.graph_id == graph_id && profile.version == version) {
                copy_profile(profile, out);
                set_report(report, DITTO_OK, "");
                return DITTO_OK;
            }
        }
        return fail(report, DITTO_NOT_FOUND, "unknown stored compressor ref");
    } catch (const std::exception& e) {
        return fail(report, DITTO_INTERNAL_ERROR, e.what());
    }
}

ditto_context* ditto_context_create(const ditto_context_options* options,
                                    ditto_report* report) {
    try {
        const uint32_t workers = options && options->worker_threads
            ? options->worker_threads
            : std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        const uint32_t limit = options && options->candidate_cache_entries
            ? options->candidate_cache_entries : 512;
        auto* context = new ditto_context(workers, limit);
        set_report(report, DITTO_OK, "");
        return context;
    } catch (const std::exception& e) {
        fail(report, DITTO_INTERNAL_ERROR, e.what());
        return nullptr;
    }
}

void ditto_context_destroy(ditto_context* context) { delete context; }

int ditto_candidates_prefetch(ditto_context* context, const int32_t* tokens,
                              size_t ntokens, int32_t min_match,
                              ditto_report* report) {
    try {
        if (!context || (!tokens && ntokens) || min_match <= 0)
            return fail(report, DITTO_INVALID_ARGUMENT,
                        "invalid context, tokens, or min_match");
        context->submit(tokens, ntokens, min_match);
        set_report(report, DITTO_OK, "");
        return DITTO_OK;
    } catch (const std::exception& e) {
        return fail(report, DITTO_INTERNAL_ERROR, e.what());
    }
}

int ditto_candidates_get(ditto_context* context, const int32_t* tokens,
                         size_t ntokens, int32_t min_match, int32_t* output,
                         size_t output_count, ditto_report* report) {
    try {
        if (!context || (!tokens && ntokens) || min_match <= 0)
            return fail(report, DITTO_INVALID_ARGUMENT,
                        "invalid context, tokens, or min_match");
        const size_t needed = ntokens * kCandidateCount;
        if ((!output && needed) || output_count < needed)
            return fail(report, DITTO_BUFFER_TOO_SMALL,
                        "candidate output buffer is too small");
        uint64_t event;
        auto future = context->submit(tokens, ntokens, min_match, &event);
#ifdef FALSE_WAITED
        perfmark_event_waited(event, 1); // Deliberately premature annotation.
#endif
        const auto result = future.get();
#ifndef FALSE_WAITED
        perfmark_event_waited(event, 1);
#endif
        std::copy(result.begin(), result.end(), output);
        set_report(report, DITTO_OK, "");
        return DITTO_OK;
    } catch (const std::exception& e) {
        return fail(report, DITTO_INTERNAL_ERROR, e.what());
    }
}

int ditto_cuda_launch_many(uintptr_t stream,
                           const ditto_cuda_launch* launches,
                           size_t launch_count, ditto_report* report) {
    try {
        if ((!launches && launch_count) || !stream)
            return fail(report, DITTO_INVALID_ARGUMENT,
                        "invalid CUDA stream or launch array");
        auto& driver = cuda_driver();
        for (size_t i = 0; i < launch_count; ++i) {
            const auto& item = launches[i];
            if (!item.function || !item.grid[0] || !item.block[0] ||
                (!item.argument_values && item.argument_count))
                return fail(report, DITTO_INVALID_ARGUMENT,
                            "invalid CUDA launch descriptor at index " +
                            std::to_string(i));
            std::vector<void*> parameters(item.argument_count);
            for (size_t j = 0; j < item.argument_count; ++j)
                parameters[j] = const_cast<uint64_t*>(
                    &item.argument_values[j]);
            const CUresult status = driver.launch(
                reinterpret_cast<CUfunction>(item.function),
                item.grid[0], item.grid[1] ? item.grid[1] : 1,
                item.grid[2] ? item.grid[2] : 1,
                item.block[0], item.block[1] ? item.block[1] : 1,
                item.block[2] ? item.block[2] : 1,
                item.shared_memory_bytes,
                reinterpret_cast<CUstream>(stream), parameters.data(), nullptr);
            if (status != CUDA_SUCCESS)
                return fail(report, DITTO_INTERNAL_ERROR,
                            "CUDA launch " + std::to_string(i) + " failed: " +
                            driver.message(status));
        }
        set_report(report, DITTO_OK, "");
        return DITTO_OK;
    } catch (const std::exception& e) {
        return fail(report, DITTO_INTERNAL_ERROR, e.what());
    }
}

}  // extern "C"
