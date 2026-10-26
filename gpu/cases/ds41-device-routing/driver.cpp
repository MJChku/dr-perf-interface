#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
using namespace ds41::gx_fixture;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

static int initialize_fixtures(Driver& api, Function kernel) {
    int cases = 0;
    for (std::uint32_t kind = 0; kind < 9; ++kind) {
        for (std::uint64_t first : {0ull, 4294967313ull}) {
            std::uint32_t count = 137;
            std::uint64_t id = ds41::synthetic::tensor_id(kind == 2 ? "layers.14.ffn.experts.37.w1.weight.scale" : "layers.14.ffn.experts.37.w1.weight");
            std::int32_t exponent = ds41::synthetic::scale_exponent(5120);
            std::vector<std::uint8_t> expected(count + 64, 0xA7), initial = expected;
            for (std::uint32_t i = 0; i < count; ++i)
                expected[32 + i] = ds41::synthetic::byte(kind,id,first+i,exponent);
            // Host recipe has separately passed the independent Python oracle.
            Buffer<std::uint8_t> output(api, initial);
            auto ptr = output.ptr + 32;
            void* args[] = {&ptr, &id, &first, &count, &kind, &exponent};
            check(api.launch(kernel, 2, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
            check(api.sync());
            require(output.read() == expected, "payload initialization byte/guard mismatch");
            ++cases;
        }
    }
    return cases;
}

static int matrix_fixtures(Driver& api,Function kernel){
    int cases=0;
    for(unsigned kind=0;kind<9;++kind)for(std::uint64_t first:{37ull,4294967313ull}){
        unsigned rows=3,columns=37,stride=43;std::uint64_t global_stride=199;
        const auto identity=ds41::synthetic::tensor_id("rectangular.shard.weight");auto id=identity;
        int exponent=ds41::synthetic::scale_exponent(2304);
        std::vector<std::uint8_t> initial(rows*stride+64,0xa7),expected=initial;
        for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<columns;++c)expected[32+r*stride+c]=ds41::synthetic::byte(kind,id,first+r*global_stride+c,exponent);
        Buffer<std::uint8_t> output(api,initial);auto pointer=output.ptr+32;
        void* args[]={&pointer,&id,&first,&rows,&columns,&stride,&global_stride,&kind,&exponent};
        check(api.launch(kernel,1,1,1,256,1,1,0,nullptr,args,nullptr));check(api.sync());
        require(output.read()==expected,"rectangular shard initializer byte/guard mismatch");++cases;
    }
    return cases;
}
static int routing_fixtures(Driver& api, Function kernel) {
    std::uint32_t rows = 5, input_stride = 389, output_stride = 9;
    constexpr int guard = 32;
    std::vector<float> logits(rows * input_stride + 2 * guard, -12345.0f);
    std::vector<float> bias(384 + 2 * guard, -12345.0f);
    std::mt19937 rng(4134);
    for (int i = 0; i < 384; ++i) bias[guard + i] = static_cast<float>(int(rng() % 101) - 50) / 37.0f;
    std::vector<int> expected_ids(rows * output_stride + 2 * guard, -778);
    std::vector<float> expected_weights(rows * output_stride + 2 * guard, -12345.0f);
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::vector<double> scores(384);
        std::vector<int> order(384);
        std::iota(order.begin(), order.end(), 0);
        for (int e = 0; e < 384; ++e) {
            float x = static_cast<float>(int(rng() % 16001) - 8000) / 100.0f;
            if (row == 1) x = 0.0f; // bias must affect selection but not weights
            if (row == 2) x = -80.0f + e * 0.01f;
            if (row == 3) x = 1000.0f + e;
            if (row == 4) x = e < 6 ? 12.0f + e : -40.0f;
            logits[guard + row * input_stride + e] = x;
            const double v = x;
            scores[e] = std::sqrt(v > 20 ? v : std::log1p(std::exp(v)));
        }
        // Independent double-precision oracle fully sorts all expert scores.
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            // Match the specified FP32 selection-score rounding, while the
            // raw-score/weight oracle retains double precision.
            const float sa = static_cast<float>(scores[a]) + bias[guard + a];
            const float sb = static_cast<float>(scores[b]) + bias[guard + b];
            return sa > sb;
        });
        double total = 1e-20;
        for (int k = 0; k < 6; ++k) total += scores[order[k]];
        for (int k = 0; k < 6; ++k) {
            expected_ids[guard + row * output_stride + k] = order[k];
            expected_weights[guard + row * output_stride + k] = scores[order[k]] / total * 1.5;
        }
    }
    Buffer<float> dl(api, logits), db(api, bias);
    Buffer<std::int32_t> ids(api, std::vector<std::int32_t>(expected_ids.size(), -778));
    Buffer<float> weights(api, std::vector<float>(expected_weights.size(), -12345.0f));
    auto l = dl.ptr + guard * sizeof(float), b = db.ptr + guard * sizeof(float);
    auto i = ids.ptr + guard * sizeof(std::int32_t), w = weights.ptr + guard * sizeof(float);
    void* args[] = {&l, &b, &i, &w, &rows, &input_stride, &output_stride};
    check(api.launch(kernel, (rows + 3) / 4, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    check(api.sync());
    require(ids.read() == expected_ids, "route selection or guard mismatch");
    const auto actual = weights.read();
    for (std::size_t n = 0; n < actual.size(); ++n) {
        const double tolerance = 2e-6 + 2e-5 * std::abs(expected_weights[n]);
        require(std::isfinite(actual[n]) && std::abs(actual[n] - expected_weights[n]) <= tolerance,
                "routing weight/stride/guard mismatch");
    }
    require(dl.read() == logits && db.read() == bias, "routing input mutated");
    // All ties, including the selection bias: deterministic smaller-ID rule.
    std::fill(logits.begin() + guard, logits.end() - guard, 0.0f);
    std::fill(bias.begin() + guard, bias.end() - guard, 0.0f);
    check(api.upload(dl.ptr, logits.data(), logits.size() * sizeof(float)));
    check(api.upload(db.ptr, bias.data(), bias.size() * sizeof(float)));
    check(api.launch(kernel, (rows + 3) / 4, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    check(api.sync());
    auto tied_ids = ids.read(); auto tied_weights = weights.read();
    for (std::uint32_t row = 0; row < rows; ++row)
        for (int k = 0; k < 6; ++k) {
            require(tied_ids[guard + row * output_stride + k] == k, "tie policy failed");
            require(std::abs(tied_weights[guard + row * output_stride + k] - 0.25f) < 2e-6,
                    "unbiased tied weights failed");
        }
    return rows + 1;
}

int main(int argc, char** argv) {
    try {
        const bool native = prepare_execution(argc, argv);
        std::ifstream input(argv[1]);
        require(input.good(), "PTX missing");
        std::string ptx{std::istreambuf_iterator<char>(input), {}};
        Driver api;
        check(api.init(0));
        Context context; check(api.create(&context, 0, 0));
        Module module; check(api.module(&module, ptx.c_str()));
        Function initialize, matrix,route;
        check(api.function(&initialize, module, "ds41_initialize_payload"));
        check(api.function(&matrix,module,"ds41_initialize_matrix"));
        check(api.function(&route, module, "ds41_route_from_logits"));
        const auto generated = initialize_fixtures(api, initialize)+matrix_fixtures(api,matrix);
        const auto routed = routing_fixtures(api, route);
        check(api.unload(module)); check(api.destroy(context));
        std::cout << "{\"numerical_status\":\"pass\",\"initializer_cases\":" << generated
                  << ",\"routing_cases\":" << routed << ",\"physical_gpu_used\":"
                  << (native ? "true" : "false") << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
