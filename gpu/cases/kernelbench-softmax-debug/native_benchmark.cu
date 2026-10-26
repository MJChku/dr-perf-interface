#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define softmax_forward_kernel_vec4 baseline_kernel
#include "synchronized.cu"
#undef softmax_forward_kernel_vec4
#define softmax_forward_kernel_vec4 optimized_kernel
#include "after.cu"
#undef softmax_forward_kernel_vec4

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void launch(int variant, const float* in, float* out, int rows, int cols) {
    if (variant == 0) baseline_kernel<<<rows, 1024, 4096>>>(in, out, cols);
    else optimized_kernel<<<rows, 1024, 4096>>>(in, out, cols);
    check(cudaGetLastError());
}
static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end()); return x[x.size()/2];
}
static void array(const std::vector<double>& x) {
    std::cout << '[';
    for (size_t i=0; i<x.size(); ++i) std::cout << (i ? "," : "") << x[i];
    std::cout << ']';
}

int main() { try {
    check(cudaSetDevice(0));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
    cudaFuncAttributes ba{}, aa{};
    check(cudaFuncGetAttributes(&ba, baseline_kernel));
    check(cudaFuncGetAttributes(&aa, optimized_kernel));
    int driver=0, runtime=0;
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::cout << std::setprecision(10) << "{\"gpu\":\"" << prop.name
              << "\",\"sm\":" << prop.major*10+prop.minor
              << ",\"driver\":" << driver << ",\"runtime\":" << runtime
              << ",\"registers\":[" << ba.numRegs << ',' << aa.numRegs
              << "],\"block_threads\":1024,\"shared_bytes\":4096,\"rounds\":11,"
                 "\"repetitions_per_round\":30,\"cases\":[\n";
    bool first=true;
    for (int cols : {256, 4096, 32768, 65536}) {
        const int rows=4096, reps=30;
        const size_t n=size_t(rows)*cols, bytes=n*sizeof(float);
        std::cerr << "Preparing " << rows << 'x' << cols << '\n';
        std::vector<float> input(n), reference(n), output(n+16);
        std::mt19937 rng(1234+cols);
        std::normal_distribution<float> normal(0, 1);
        for (auto& v : input) v=normal(rng);
        #pragma omp parallel for
        for (int r=0; r<rows; ++r) {
            const float* row=input.data()+size_t(r)*cols;
            double mx=*std::max_element(row, row+cols), total=0;
            for (int j=0; j<cols; ++j) total+=std::exp(double(row[j])-mx);
            for (int j=0; j<cols; ++j)
                reference[size_t(r)*cols+j]=float(std::exp(double(row[j])-mx)/total);
        }
        float *in=nullptr, *out=nullptr;
        check(cudaMalloc(&in, bytes)); check(cudaMalloc(&out, bytes+16*sizeof(float)));
        check(cudaMemcpy(in, input.data(), bytes, cudaMemcpyHostToDevice));
        std::fill(output.begin(), output.end(), -12345.f);
        check(cudaMemcpy(out, output.data(), bytes+16*sizeof(float), cudaMemcpyHostToDevice));
        double errors[2]={};
        // Verify every output element before timing; the CPU reference uses
        // double arithmetic and is rounded to the FP32 output format.
        for (int variant=0; variant<2; ++variant) {
            launch(variant, in, out, rows, cols);
            check(cudaMemcpy(output.data(), out, bytes+16*sizeof(float), cudaMemcpyDeviceToHost));
            double largest=0; unsigned long long bad=0;
            #pragma omp parallel for reduction(max:largest) reduction(+:bad)
            for (size_t i=0; i<n; ++i) {
                double error=std::abs(double(output[i])-reference[i]);
                largest=std::max(largest,error);
                if (!std::isfinite(output[i]) || error>2e-7+2e-5*std::abs(reference[i])) ++bad;
            }
            for (size_t i=n; i<n+16; ++i) if (output[i]!=-12345.f) ++bad;
            if (bad) throw std::runtime_error("native numerical/guard check failed");
            errors[variant]=largest;
        }
        // Same stream, buffers, inputs, block size and shared memory. Alternate
        // execution order to reduce systematic clock/thermal ordering bias.
        for (int i=0; i<30; ++i) { launch(0,in,out,rows,cols); launch(1,in,out,rows,cols); }
        check(cudaDeviceSynchronize());
        cudaEvent_t start, stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        std::vector<double> times[2];
        for (int round=0; round<11; ++round) for (int position=0; position<2; ++position) {
            int variant=(round+position)%2;
            check(cudaEventRecord(start));
            for (int i=0; i<reps; ++i) launch(variant,in,out,rows,cols);
            check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
            float ms=0; check(cudaEventElapsedTime(&ms,start,stop));
            times[variant].push_back(double(ms)*1000/reps);
        }
        std::cout << (first ? "" : ",\n") << "{\"rows\":" << rows << ",\"columns\":" << cols
                  << ",\"baseline_us\":" << median(times[0])
                  << ",\"optimized_us\":" << median(times[1])
                  << ",\"latency_reduction_percent\":" << 100*(1-median(times[1])/median(times[0]))
                  << ",\"speedup\":" << median(times[0])/median(times[1])
                  << ",\"baseline_rounds_us\":";
        array(times[0]); std::cout << ",\"optimized_rounds_us\":"; array(times[1]);
        std::cout << ",\"baseline_max_abs_error\":" << errors[0]
                  << ",\"optimized_max_abs_error\":" << errors[1]
                  << ",\"all_elements_checked\":true,\"passed\":true}" << std::flush;
        first=false;
        std::cerr << "Measured " << rows << 'x' << cols << ": " << median(times[0])
                  << " -> " << median(times[1]) << " us\n";
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        check(cudaFree(in)); check(cudaFree(out));
    }
    std::cout << "\n]}\n";
    check(cudaDeviceReset());
    return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
