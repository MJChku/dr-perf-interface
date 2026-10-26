#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) { try {
    if (argc != 4) return 2;
    drperf::gpu::Module module(argv[1]);
    bool heldout = std::string(argv[3]) == "heldout";
    std::vector<int> widths = heldout ? std::vector<int>{4, 252, 4100, 8192, 65536, 65540}
                                     : std::vector<int>{256, 1024, 4096};
    if (std::string(argv[3]) == "fallback") widths = {65540};
    if (std::string(argv[3]) == "original-width") widths = {65536};
    double max_error = 0;
    int cases = 0;
    for (int rows : {1, 2}) for (int cols : widths) for (int pattern = 0; pattern < 3; ++pattern) {
        size_t count = size_t(rows) * cols;
        drperf::gpu::Buffer<float> input(count), output(count + 16, -12345.f);
        std::mt19937 gen(1234 + rows * 7 + cols + pattern);
        std::uniform_real_distribution<float> random(-10, 10);
        for (size_t i=0; i<count; ++i)
            input[i] = pattern == 0 ? random(gen) : pattern == 1 ? 17.f :
                       (i % cols == size_t(cols / 2) ? 100.f : -100.f);
        std::vector<float> original(input.begin(), input.end());
        auto* in = input.data(); auto* out = output.data();
        void* args[] = {&in, &out, &cols};
        {
            drperf::gpu::Region region(argv[2], {{"elements", int64_t(count)}, {"rows", rows}});
            module.launch("_Z27softmax_forward_kernel_vec4PKfPfi", rows, 1024, args, 4096);
        }
        for (int r=0; r<rows; ++r) {
            double mx = *std::max_element(input.begin()+r*cols, input.begin()+(r+1)*cols);
            double sum = 0;
            for (int j=0; j<cols; ++j) sum += std::exp(double(input[r*cols+j])-mx);
            double out_sum = 0;
            for (int j=0; j<cols; ++j) {
                double expected = std::exp(double(input[r*cols+j])-mx) / sum;
                double actual = output[r*cols+j];
                double error = std::abs(actual-expected);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual) || error > 2e-7 + 2e-5*expected) {
                    std::cerr << "rows=" << rows << " cols=" << cols << " pattern=" << pattern
                              << " index=" << r*cols+j << " expected=" << expected << " actual=" << actual << '\n';
                    throw std::runtime_error("softmax numerical oracle failed");
                }
                out_sum += actual;
            }
            if (std::abs(out_sum-1) > 2e-5) throw std::runtime_error("normalization failed");
        }
        if (!std::equal(input.begin(), input.end(), original.begin())) throw std::runtime_error("input modified");
        for (size_t i=count; i<output.size(); ++i)
            if(output[i] != -12345.f) throw std::runtime_error("output guard overwritten");
        ++cases;
    }
    std::cout << "{\"suite\":\"" << argv[3] << "\",\"cases\":" << cases
              << ",\"max_absolute_error\":" << max_error << ",\"passed\":true}\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
