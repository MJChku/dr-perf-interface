#include "../src4f5/third_party/vllm-385dce36/cooperative_topk.cuh"
template __global__ void vllm::cooperative::cooperative_topk_cs16<2048>(vllm::cooperative::CooperativeTopKParams<2048>);
extern "C" __global__ void topk_shared(int*out){out[0]=vllm::cooperative::kSmemSize8;out[1]=sizeof(vllm::cooperative::Smem16);}
