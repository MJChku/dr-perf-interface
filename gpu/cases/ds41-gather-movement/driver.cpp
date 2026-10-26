#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "ds41/gx_fixture.hpp"

using namespace ds41::gx_fixture;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Shape { std::uint32_t queries, keys, width, rows; bool invalid; bool aligned=false; };
static void fixture(Driver& api, Function fn, Shape shape, unsigned seed) {
    auto [queries, keys, width, rows, invalid, aligned] = shape;
    const std::uint64_t cs = width + (aligned?0:7), is = keys + 3;
    const std::uint64_t os = static_cast<std::uint64_t>(keys) * width + (aligned?0:11);
    constexpr std::size_t guard = 32;
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> cache(guard * 2 + rows * cs, 0xA7);
    for (std::size_t i = guard; i < cache.size() - guard; ++i) cache[i] = rng() % 256;
    std::vector<std::int32_t> indices(guard * 2 + queries * is, -778);
    std::vector<std::int32_t> expected_errors(guard * 2 + queries, 0);
    std::fill(expected_errors.begin(), expected_errors.begin() + guard, -778);
    std::fill(expected_errors.end() - guard, expected_errors.end(), -778);
    auto initial_errors = expected_errors;
    std::vector<std::uint8_t> expected(guard * 2 + queries * os, 0xCD);
    const auto initial = expected;
    // Independent nested CPU oracle; includes repeats, masks, permutation,
    // first/last rows, byte extremes, invalid negatives and positive bounds.
    for (std::uint32_t q = 0; q < queries; ++q) {
        for (std::uint32_t k = 0; k < keys; ++k) {
            int row = static_cast<int>(rng() % rows);
            if (k % 7 == 0) row = -1;
            if (k == 1) row = 0;
            if (k == 2) row = static_cast<int>(rows - 1);
            if (invalid && k == 3) row = static_cast<int>(rows);
            if (invalid && k == 4) row = -2;
            indices[guard + q * is + k] = row;
            if (row != -1 && (row < 0 || static_cast<std::uint32_t>(row) >= rows))
                expected_errors[guard + q] = 1;
            for (std::uint32_t d = 0; d < width; ++d)
                expected[guard + q * os + k * width + d] =
                    row >= 0 && static_cast<std::uint32_t>(row) < rows ? cache[guard + row * cs + d] : 0;
        }
    }
    Buffer<std::uint8_t> dc(api, cache), output(api, initial);
    Buffer<std::int32_t> di(api, indices), errors(api, initial_errors);
    DevicePtr c = dc.ptr + guard, idx = di.ptr + guard * sizeof(std::int32_t);
    DevicePtr o = output.ptr + guard, e = errors.ptr + guard * sizeof(std::int32_t);
    auto cache_stride = cs, index_stride = is, out_stride = os;
    void* args[] = {&c, &idx, &o, &e, &queries, &keys, &width, &rows,
                    &cache_stride, &index_stride, &out_stride};
    // Zero work avoids zero-sized grids; these cases verify untouched buffers.
    if (queries && keys && width) {
        unsigned grid = (static_cast<std::uint64_t>(keys) * width + 127) / 128;
        check(api.launch(fn, grid, queries, 1, 128, 1, 1, 0, nullptr, args, nullptr));
        check(api.sync());
    }
    require(output.read() == expected, "gather output/padding/guard mismatch");
    require(errors.read() == expected_errors, "error flag/guard mismatch");
    require(dc.read() == cache, "input cache modified");
    require(di.read() == indices, "input indices modified");
}
int main(int argc, char** argv) {
    try {
        const bool native = prepare_execution(argc, argv);
        std::ifstream input(argv[1]);
        require(input.good(), "PTX input missing");
        std::string ptx{std::istreambuf_iterator<char>(input), {}};
        Driver api;
        check(api.init(0));
        Context context;
        check(api.create(&context, 0, 0)); // this process owns exactly one device context
        Module module;
        check(api.module(&module, ptx.c_str()));
        Function fn;
        check(api.function(&fn, module, "ds41_gather_bytes"));
        unsigned count = 0;
        for (auto s : {Shape{1, 1, 1, 1, false}, Shape{3, 9, 17, 13, false},
                       Shape{2, 7, 64, 19, true}, Shape{1, 6, 512, 37, false},
                       Shape{3,37,160,97,true,true},Shape{2,67,5120,131,false,true},
                       Shape{4, 0, 17, 7, false}, Shape{0, 4, 9, 3, false},
                       Shape{2, 3, 0, 5, false}}) {
            fixture(api, fn, s, 9341 + count);
            ++count;
        }
        check(api.unload(module));
        check(api.destroy(context));
        std::cout << "{\"numerical_status\":\"pass\",\"kernel\":\"ds41_gather_bytes\","
                     "\"cases\":" << count << ",\"tolerance\":\"bitwise byte equality\","
                     "\"physical_gpu_used\":" << (native ? "true" : "false") << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
