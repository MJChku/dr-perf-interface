#include "drperf_gpu.h"
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    drperf::gpu::Module module(argv[1]);
    for (const char* kernel : {"predicated_pair", "branched_pair"}) {
        for (unsigned n : {8u, 16u, 32u, 48u, 64u}) {
            std::vector<float> input(64), output(66, -777);
            for (unsigned i = 0; i < 64; ++i) input[i] = float(i + 1);
            auto in = input.data(), out = output.data() + 1;
            void* args[] = {&in, &out, &n};
            {
                drperf::gpu::Region region(kernel, {{"n", n}});
                module.launch(kernel, 1, 64, args);
            }
            for (unsigned i = 0; i < 64; ++i)
                if (out[i] != (i < n ? input[i] : 0))
                    throw std::runtime_error("basic-block numerical mismatch");
            if (output.front() != -777 || output.back() != -777)
                throw std::runtime_error("basic-block guard overwritten");
        }
    }
}
