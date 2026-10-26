#include "drperf_gpu_device.cuh"
using drperf::gpu::DeviceRegion;
using drperf::gpu::DeviceScope;

extern "C" __device__ __noinline__ void annotated_adds(float* value,int iterations) {
    DeviceRegion inner("loop", {{"iterations",iterations}});
    float x=*value;
    #pragma unroll 1
    for(int j=0;j<iterations;++j)x=__fadd_rn(x,1.f);
    *value=x;
}

extern "C" __global__ void device_sections(const float* input,float* output,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    // Device-computed, data-dependent PCV; not supplied by a host marker.
    const int iterations=i<n?int(input[i])%3+1:0;
    DeviceRegion outer("outer", {{"active",int(i<n)}});
    if(i<n) {
        float x=__fmul_rn(input[i],2.f);
        annotated_adds(&x,iterations);
        output[i]=x;
    }
}

extern "C" __global__ void cooperative_sections(float* output,int iterations) {
    const int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;
    DeviceRegion block("cooperative",DeviceScope::block, {{"iterations",iterations}, {"lanes",blockDim.x}});
    __shared__ float shared[64];
    float x=float(i);
    {
        DeviceRegion warp("cooperative",DeviceScope::warp, {{"iterations",iterations}, {"lanes",min(32,int(blockDim.x)-(t/32)*32)}});
        #pragma unroll 1
        for(int j=0;j<iterations;++j)x=__fadd_rn(x,1.f);
    }
    shared[t]=x;
    __syncthreads();
    output[i]=__fmul_rn(shared[(t+1)%blockDim.x],2.f);
}

extern "C" __global__ void repeated_sections(float* output,int rounds) {
    float x=float(threadIdx.x);
    #pragma unroll 1
    for(int j=0;j<rounds;++j) {
        DRPERF_GPU_DEVICE_BEGIN("repeated",DeviceScope::block, {"round",j});
        x=__fadd_rn(x,1.f);
        DRPERF_GPU_DEVICE_END();
    }
    output[threadIdx.x]=x;
}

extern "C" __global__ void invalid_sections(float* output,int mode) {
    if(mode==0) { DRPERF_GPU_DEVICE_END();return; }
    const auto scope=mode==1?DeviceScope(9):mode==2?DeviceScope::lane:DeviceScope::block;
    if(mode==3&&threadIdx.x!=0)return;
    drperf::gpu::device_begin("invalid",scope);
    drperf::gpu::device_pcv("n",mode==4?int(threadIdx.x):1);
    if(mode==5&&threadIdx.x==0)drperf::gpu::device_pcv("extra",1);
    if(mode==6)drperf::gpu::device_pcv("n",1);
    if(mode!=8)drperf::gpu::device_ready();
    output[threadIdx.x]=1.f;
    if(mode==7)drperf::gpu::device_pcv("late",1);
    if(mode!=2)DRPERF_GPU_DEVICE_END();
}

// Reduced sparse-attention pattern: math scales with selected K, whereas
// clearing a dense probability vector scales with total history T.
extern "C" __global__ void sparse_regions(float* output,int history,int selected,int explain_history) {
    const int t=threadIdx.x;
    {
        DeviceRegion r("dense_clear",DeviceScope::block, explain_history
            ? std::initializer_list<drperf::gpu::DevicePCV>{{"selected",selected},{"history",history}}
            : std::initializer_list<drperf::gpu::DevicePCV>{{"selected",selected}}); // incomplete first explanation
        #pragma unroll 1
        for(int i=t;i<history;i+=blockDim.x)output[i]=0.f;
    }
    {
        DeviceRegion r("selected_math",DeviceScope::block, {{"selected",selected}});
        float x=0.f;
        #pragma unroll 1
        for(int i=t;i<selected;i+=blockDim.x)x=__fadd_rn(x,1.f);
        output[history+t]=x;
    }
}
